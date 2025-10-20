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
import copy


class TDSRAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Dict[str, Any] = nonpytree_field()
    ex_actions: Any = nonpytree_field()

    def fb_loss(self, batch, grad_params, module_name="value"):
        batch_size = batch["observations"].shape[0]
        obs_shape = batch["observations"].shape[1:-1]
        obs = batch["observations"].reshape(
            (batch_size, *obs_shape, self.config["frame_stack"], -1)
        )
        next_obs = batch["next_observations"].reshape(
            (batch_size, *obs_shape, self.config["frame_stack"], -1)
        )
        phi, psi = self.network.select(module_name)(
            obs[..., 0, :],
            next_obs.transpose(0, len(obs_shape)+1, *range(1,len(obs_shape)+1), -1).reshape(-1, *obs_shape, next_obs.shape[-1]),
            actions=batch["actions"] if self.config["action_forward"] else None,
            info=True,
            params=grad_params,
        )
        phi_target, psi_target = self.network.select(f"target_{module_name}")(
            next_obs[..., -1, :],
            next_obs[..., 0, :],
            actions=batch['next_actions'] if self.config['action_forward'] else None,
            info=True,
            params=grad_params,
        )
        if self.config['loss_type'] == "bdino":
            # softmax both z_pred and z_target
            eps = 1e-6
            # tps = 1 # temperature
            tps = self.config['pred_loss_temp_s']
            tpt = self.config['pred_loss_temp_t']
            phi = jnp.log(jax.nn.softmax(phi / tps, axis=-1) + eps)
            psi = jax.nn.softmax((psi) / tpt, axis=-1) 
            phi_target = jnp.log(jax.nn.softmax(phi_target / tps, axis=-1) + eps)
            psi_target = jax.nn.softmax((psi_target) / tpt, axis=-1) 
        else:
            assert self.config['loss_type'] == "fb", f"Loss type {self.config['loss_type']} is not supported"


        phi_target = jax.lax.stop_gradient(phi_target)
        psi_target = jax.lax.stop_gradient(psi_target)

        if len(phi.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi = psi[None, ...]
        psi = psi.reshape(
            psi.shape[0], batch_size, self.config["frame_stack"], psi.shape[-1]
        )
        fb = jnp.einsum("eik,ejk->ije", phi, psi[:, :, 0, :])
        fb_target = jnp.einsum("eik,ejk->ije", phi_target, psi_target)

        assert fb.shape[0] == batch_size and fb.shape[1] == batch_size
        I = jnp.eye(batch_size)
        ones = jnp.ones_like(I)

        fb_offdiag_loss = (
            fb - self.config["discount"] ** self.config["n_step"] * fb_target
        ) * (ones - I)[..., None]
        discount = self.config["discount"] ** jnp.arange(self.config["n_step"])
        fb_diag_loss = jnp.einsum("eik,eijk->ije", phi, psi)
        fb_diag_loss = jnp.einsum("ije,j->ie", fb_diag_loss, discount)

        fb_offdiag_loss = jnp.sum(jnp.square(fb_offdiag_loss)) / (
            batch_size * (batch_size - 1)
        )
        fb_diag_loss = jnp.mean(fb_diag_loss)
        fb_loss = 0.5 * fb_offdiag_loss - fb_diag_loss

        covb = jnp.einsum("eik,ejk->ije", psi[:, :, 0, :], psi[:, :, 0, :])
        ortho_loss_diag = -2 * jnp.mean(I[..., None] * covb)
        ortho_loss_offdiag = jnp.mean(jnp.square((ones - I)[..., None] * covb))
        ortho_loss = ortho_loss_diag + ortho_loss_offdiag
        total_loss = fb_loss + self.config["ortho_coef"] * ortho_loss

        return total_loss, {
            "fb_loss": fb_loss,
            "ortho_loss": ortho_loss,
            "total_loss": total_loss,
        }
    
    def actor_loss(self, batch, grad_params, rng=None):
        batch_size = batch["observations"].shape[0]
        obs_shape = batch["observations"].shape[1:-1]
        phi, psi = self.network.select("value")(
            batch["observations"].reshape(batch_size, *obs_shape, self.config["n_step"], -1)[..., -1, :],
            batch["actor_goals"].reshape(batch_size, *obs_shape, self.config["n_step"], -1)[..., -1, :],
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 3:
            phi = jnp.mean(phi, axis=0)
        if len(psi.shape) == 3:
            psi = jnp.mean(psi, axis=0)

        dist = self.network.select("actor")(phi, psi, params=grad_params)
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

        critic_loss, critic_info = self.fb_loss(batch, grad_params, "value")
        for k, v in critic_info.items():
            info[f"value/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = self.config["alignment"] * critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "value")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        obs_shape = observations.shape[0:-1]
        phi, psi = self.network.select("value")(
            observations.reshape(*obs_shape, self.config["n_step"], -1)[..., -1, :],
            goals.reshape(*obs_shape, self.config["n_step"], -1)[..., -1, :],
            info=True,
        )
        phi = jnp.mean(phi, axis=0)
        phi = jax.lax.stop_gradient(phi)
        psi = jnp.mean(psi, axis=0)
        psi = jax.lax.stop_gradient(psi)

        dist = self.network.select("actor")(
            phi, psi, temperature=temperature, goal_encoded=True
        )
        actions = dist.sample(seed=seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls, seed, ex_observations, ex_actions, config, use_same_val_critic=True
    ):
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
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoder = encoder_module()
            encoders["value_state"] = encoder
            encoders["value_goal"] = encoder

        if config["use_obs_latent_dim"]:
            latent_dim = ex_observations.shape[-1]
        else:
            latent_dim = config["value_latent_dim"]
        ex_obs_latent = jnp.zeros((1, latent_dim))

        value_def = GCBilinearValue(
            hidden_dims=config["value_hidden_dims"],
            latent_dim=latent_dim,
            layer_norm=config["layer_norm"],
            ensemble=True,
            value_exp=True,
            state_encoder=encoders.get("value_state"),
            goal_encoder=encoders.get("value_goal"),
            normalize_psi=config["normalize_psi"],
            compute_v=False,
            action_forward=config['action_forward']
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
            target_value=(copy.deepcopy(value_def), ex_value_inputs),
            actor=(actor_def, (ex_obs_latent, ex_obs_latent)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])

        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            ex_actions=ex_actions,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="tdsr",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(64, 64, 64),  # Value network hidden dimensions.
            value_latent_dim=64,
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            alpha=1.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            action_forward=False,  # Whether to use action-conditioning in psi.
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
            ortho_coef=1.0,
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack
            repr_stopgrad=False,
            normalize_psi=True,
            n_step=1,
            loss_type="fb",
            pred_loss_temp_s=1.0,
            pred_loss_temp_t=1.0,
            use_obs_latent_dim=True,
        )
    )
    return config
