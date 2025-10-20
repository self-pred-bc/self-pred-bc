# BZ 11.18: I heavily suspect the reason the code is not working here is the contrastive loss: I've kept the same procedure as CRL and see what we can do here:

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.networks import (
    GCActor,
    GCBilinearValue,
    GCDiscreteActor,
    GCDiscreteBilinearCritic,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from typing import Any, Dict


class TRAAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Dict[str, Any] = nonpytree_field()
    ex_actions: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name="value"):
        batch_size = batch["observations"].shape[0]

        v, phi, psi = self.network.select(module_name)(
            batch["observations"],
            batch["value_goals"],
            actions=batch['actions'] if self.config['action_forward'] else None,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi = psi[None, ...]

        if self.config['loss_type'] == "bdino":
            # softmax both z_pred and z_target
            eps = 1e-6
            # tps = 1 # temperature
            tps = self.config['pred_loss_temp_s']
            tpt = self.config['pred_loss_temp_t']
            phi = jnp.log(jax.nn.softmax(phi / tps, axis=-1) + eps)
            psi = jax.nn.softmax((psi) / tpt, axis=-1) 
        else:
            assert self.config['loss_type'] == "contrastive", f"Loss type {self.config['loss_type']} is not supported"
        logits = jnp.einsum("eik,ejk->ije", phi, psi) / jnp.sqrt(phi.shape[-1])

        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.

        # binary NCE
        # I = jnp.eye(batch_size)
        # contrastive_loss = jax.vmap(
        #     lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
        #     in_axes=-1,
        #     out_axes=-1,
        # )(logits)
        # contrastive_loss = jnp.mean(contrastive_loss)

        # symmetric infoNCE
        assert logits.shape[0] == batch_size and logits.shape[1] == batch_size
        I = jnp.eye(batch_size)

        contrastive_loss = -(
            jax.nn.log_softmax(logits, axis=0) * I[..., None]
            + jax.nn.log_softmax(logits, axis=1) * I[..., None]
        )
        contrastive_loss = jnp.mean(contrastive_loss)
        # regularization term for weight decay
        l2_reg = jnp.mean(phi ** 2) + jnp.mean(psi ** 2)
        contrastive_loss += self.config["repr_reg"] * jnp.mean(l2_reg)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            "contrastive_loss": contrastive_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
            "binary_accuracy": jnp.mean((logits > 0) == I),
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logits": logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):

        v, phi, psi = self.network.select("value")(
            batch["observations"],
            batch["actor_goals"],
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 3:
            phi = jnp.mean(phi, axis=0)
        # phi = jax.lax.stop_gradient(phi)
        if len(psi.shape) == 3:
            psi = jnp.mean(psi, axis=0)
        
        # if self.config["repr_stopgrad"]:
        #if self.config["alignment"]:
        #    phi = jax.lax.stop_gradient(phi)
        #    psi = jax.lax.stop_gradient(psi)
        dist = self.network.select("actor")(phi, psi, params=grad_params)
        log_prob = dist.log_prob(batch["actions"])

        # actor_loss = -(exp_a * log_prob).mean()
        actor_loss = -log_prob.mean()

        actor_info = {
            "actor_loss": actor_loss,
            # 'adv': adv.mean(),
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

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, "value")
        for k, v in critic_info.items():
            info[f"value/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = self.config["alignment"] * critic_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        _, phi, psi = self.network.select("value")(
            observations,
            goals,
            info=True,
        )
        phi = jnp.mean(phi, axis=0)
        phi = jax.lax.stop_gradient(phi)
        psi = jnp.mean(psi, axis=0)
        psi = jax.lax.stop_gradient(psi)

        dist = self.network.select("actor")(phi, psi, temperature=temperature, goal_encoded=True)
        actions = dist.sample(seed=seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, use_same_val_critic=True):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        ex_goals_val = ex_observations  # jnp.zeros((1, 512))
        ex_goals_act = jnp.zeros((1, config["value_latent_dim"]))
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        # latent_dim = ex_observations.shape[-1]
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            if config["image_indep_cnn"]:
                encoders["value_state"] = encoder_module()
                encoders["value_goal"] = encoder_module()
            else:
                encoder = encoder_module()
                encoders["value_state"] = encoder
                encoders["value_goal"] = encoder
            # latent_dim = encoders["value_state"].mlp_hidden_dims[-1]
        if config["use_obs_latent_dim"]:
            latent_dim = ex_observations.shape[-1]
        else:
            latent_dim = config["value_latent_dim"]
        ex_obs_latent = jnp.zeros((1, latent_dim))
        ex_obs_latent = jnp.zeros((1, latent_dim))
            # encoders["value_state"] = encoder_module()
            # encoders["value_goal"] = encoder_module()
            # encoders["actor"] = GCEncoder(concat_encoder=encoder_module())

        value_def = GCBilinearValue(
            hidden_dims=config["value_hidden_dims"],
            # latent_dim=ex_observations.shape[1],
            latent_dim = latent_dim,
            layer_norm=config["layer_norm"],
            ensemble=True,
            value_exp=True,
            state_encoder=encoders.get("value_state"),
            goal_encoder=encoders.get("value_goal"),
            action_forward=config["action_forward"],
        )

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

        if config["action_forward"]:
            ex_value_inputs = (ex_observations, ex_goals_val, ex_actions)
        else:
            ex_value_inputs = (ex_observations, ex_goals_val)
        network_info = dict(
            value=(value_def, ex_value_inputs),
            value_target=(value_def, ex_value_inputs),
            # actor=(actor_def, (ex_observations, ex_goals)),
            actor=(actor_def, (ex_obs_latent, ex_obs_latent)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config), ex_actions=ex_actions)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="tra",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            image_indep_cnn=True,  # Whether to use separate CNNs  for state/goal
            value_hidden_dims=(64, 64, 64),  # Value network hidden dimensions.
            value_latent_dim=64,
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            alpha=1.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
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
            repr_reg=1e-6,
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack
            repr_stopgrad=False,
            action_forward=False,  # Whether to use the action as input to the value network.
            loss_type='contrastive', # loss type for training value function
            pred_loss_temp_s=1.0,
            pred_loss_temp_t=1.0,
            use_obs_latent_dim=True,
        )
    )
    return config
