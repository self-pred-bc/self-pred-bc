
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.networks import (
    GCActor,
    GCDiscreteActor,
    Encoder,
    LatentPredictor,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from typing import Any, Dict
import copy
from enum import Enum, auto




class BYOLMinAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Dict[str, Any] = nonpytree_field()
    ex_actions: Any = nonpytree_field()
    

    def compute_pred_loss(self, z_pred, z_target, loss_type='l2_cos'):
        # info = {}
        if loss_type == 'l2':
            sqdiff = (z_pred - z_target) ** 2  # shape (B, latent_dim)
            loss_per_sample = jnp.sum(sqdiff, axis=-1)  # shape (B,)
            return jnp.mean(loss_per_sample)#, info           # scalar
        elif loss_type == 'l2_cos' or loss_type == 'l2_cos_sq':
            # normalized L2: first l2norm each vector, then MSE
            # Gradients of norrmalized L2 loss and cosine similiarity are proportional.
            # See: https://stats.stackexchange.com/a/146279
            eps = 1e-3
            norm_pred = z_pred / (jnp.linalg.norm(z_pred, ord=2, axis=-1, keepdims=True) + eps)
            norm_targ = z_target / (jnp.linalg.norm(z_target, ord=2, axis=-1, keepdims=True) + eps)
            sqdiff = (norm_pred - norm_targ) ** 2
            loss_per_sample = jnp.sum(sqdiff, axis=-1)
            if loss_type == 'l2_cos_sq':
                loss_per_sample = jnp.sqrt(loss_per_sample)
            return jnp.mean(loss_per_sample)#, info
        elif loss_type == 'bdino':
            # softmax both z_pred and z_target
            eps = 1e-6
            s = jax.nn.softmax(z_pred, axis=-1)  # shape (B, K)
            t = jax.nn.softmax(z_target, axis=-1)  # shape (B, K)
            loss_per_sample = -jnp.sum(t * jnp.log(s + eps), axis=-1)  # shape (B,)
            loss = jnp.mean(loss_per_sample)
            return loss #,info  # scalar

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")


    def process_input(self, phi, psi, extra):
        phi_goal = extra['phi_goal']
        return phi, phi_goal

    def pred_loss(self, observations, goals, actions, grad_params,  use_backwards=False):
        batch_size = observations.shape[0]

        current_representations = self.network.select("encoder")(observations, params=grad_params)
        predictions = self.network.select('predictor')(
            current_representations,
            actions,
            params=grad_params,
            use_backwards=use_backwards,
        )
        target_module = 'target_encoder' if self.config['target'] else 'encoder'
        future_representations = self.network.select(target_module)(goals, params=grad_params)
        future_representations = jax.lax.stop_gradient(future_representations)

        if len(current_representations.shape) == 2:  # Non-ensemble
            current_representations = current_representations[None, ...]
            future_representations = future_representations[None, ... ]

        pred_loss = self.compute_pred_loss(predictions, future_representations, loss_type=self.config['pred_loss_type']) 
        total_pred_loss = pred_loss


        info = {
            "pred_loss": pred_loss,
        }
        return total_pred_loss, info


    def actor_loss(self, batch, grad_params, rng=None):        
        phi = self.network.select("encoder")(batch["observations"], params=grad_params)
        phi_goal = self.network.select("encoder")(batch["actor_goals"], params=grad_params)

        
        if len(phi.shape) == 3:
            # average
            phi = jnp.mean(phi, axis=0)
            phi_goal = jnp.mean(phi_goal, axis=0)

        dist = self.network.select("actor")(phi, phi_goal, params=grad_params)
        log_prob = dist.log_prob(batch["actions"])

        actor_loss = -log_prob.mean()

        actor_info = {
            "actor_loss": actor_loss,
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:  # pylint: disable=unsubscriptable-object
            actor_info.update(
                {
                    "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                    "std": jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        stats = {}
        if self.config['pred_both']:
            f_pred_loss, f_pred_info = self.pred_loss(batch["observations"], batch["value_goals"], batch['actions'], grad_params)
            b_pred_loss, b_pred_info = self.pred_loss(batch["value_goals"], batch["observations"], batch['actions'], grad_params, use_backwards=True)
            for k, v in f_pred_info.items():
                info[f"pred_f/{k}"] = v
            for k, v in b_pred_info.items():
                info[f"pred_b/{k}"] = v

            pred_loss = f_pred_loss + b_pred_loss
        elif self.config['pred_backwards']:
            pred_loss, pred_info = self.pred_loss(batch["value_goals"], batch["observations"], batch['actions'], grad_params)
            for k, v in pred_info.items():
                info[f"pred_b/{k}"] = v
        else:
            pred_loss, pred_info = self.pred_loss(batch["observations"], batch["value_goals"], batch['actions'], grad_params)
            for k, v in pred_info.items():
                info[f"pred_f/{k}"] = v
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        pred_loss = self.config["alignment"] * pred_loss
        actor_loss = self.config["bc_weight"] * actor_loss
        loss = pred_loss + actor_loss
        info['stats'] = stats
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'encoder')

        # return self.replace(network=new_network, rng=new_rng), info
        return self.replace(
            network=new_network,
            rng=new_rng,
        ), info



    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        phi = self.network.select("encoder")(observations)
        phi_goals = self.network.select("encoder")(goals)

        phi = jax.lax.stop_gradient(phi)
        phi_goals = jax.lax.stop_gradient(phi_goals)

        # average
        if len(phi.shape) == 2:
            phi = jnp.mean(phi, axis=0)
            phi_goals = jnp.mean(phi_goals, axis=0)

        dist = self.network.select("actor")(phi, phi_goals, temperature=temperature, goal_encoded=True)
        actions = dist.sample(seed=seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        # Define encoders.
        image_encoder = None
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            image_encoder = encoder_module(dropout_rate=config.get('encoder_dropout'), layer_norm=config.get('encoder_layer_norm'))

        if config["use_obs_latent_dim"]:
            latent_dim = ex_observations.shape[-1]
        else:
            latent_dim = config["value_latent_dim"]

        latent_obs_dim = latent_dim
        ex_obs_latent = jnp.zeros((1, latent_obs_dim))
        ex_pred_latent = jnp.zeros((config['ensemble_size'], 1, latent_obs_dim))

        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        encoder_def = Encoder(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=latent_dim,
            layer_norm=config['layer_norm'],
            ensemble_size=config['ensemble_size'],
            image_encoder=image_encoder,
            identity_phi=False,
        )

        predictor_def  = LatentPredictor(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=latent_dim,
            layer_norm=config["layer_norm"],
            pred_both=config['pred_both'],
            action_forward=config['action_forward'],
            ensemble_size=config['ensemble_size'],
        )

        assert not (config.get('pred_both') and config['pred_backwards']), 'only one enabled'

        if config["discrete"]:
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
            )

        network_info = dict(
            # value=(value_def, (ex_observations, ex_goals_val, False, config['pred_both'], False, ex_actions)),
            encoder = (encoder_def, (ex_observations,)),
            target_encoder = (copy.deepcopy(encoder_def), (ex_observations,)),
            predictor=(predictor_def, (ex_pred_latent, ex_actions, False, config['pred_both'])),
            actor=(actor_def, (ex_obs_latent, ex_obs_latent)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])

        
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_encoder'] = params['modules_encoder']
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config), ex_actions=ex_actions)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="byol_min",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(64, 64, 64),  # Value network hidden dimensions.
            value_latent_dim=64,
            use_obs_latent_dim=True, # use observation as latent dim shape, enabled by default to match TRA
            layer_norm=True,  # Whether to use layer normalization.
            encoder_layer_norm=False, # Whether to use layer normalization in the (CNN) encoder
            action_forward=False, # Whether to use action-conditioning in forward prediction
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target=False, # whether to use a target, e.g. False behaves like tau=1.0
            alpha=1.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            normalize_phi=False, # Whether to make phi a unit vector
            # Encoder hyperparameters
            ensemble_size=2,
            # BYOL hyerparameters
            pred_backwards=False,
            pred_both=False,
            pred_loss_type='l2_cos',
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            encoder_dropout=0.0,
            # encoder_blocks=1,
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            alignment=1.0,
            bc_weight=1.0,
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack
            repr_stopgrad=False,
        )
    )
    return config
