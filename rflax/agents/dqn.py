"""Implementation of vanilla DQN. (TODO: support advanced)"""

from flax.core.frozen_dict import FrozenDict
from ml_collections import ConfigDict
from optax import adam
from rflax.agents import DiscreteAgent, TargetParams
from rflax.components.nets.value import StateDiscreteActionValue
from rflax.components.loss import q_learning_loss
from rflax.utils import init_model
from rflax.components.initializers import kernel_default, bias_default
from rflax.types import Array, MetricDict, PRNGKey, VariableDict, ConfigDictLike

import chex
import jax
import jax.numpy as jnp
from functools import partial
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from typing import Tuple, Optional


@partial(jax.jit, static_argnames=("enable_dropout",))
def _critic_apply(rng: PRNGKey, critic: TrainState, observations: Array,
                  enable_dropout: bool) -> chex.ArrayDevice:
  q_vals = critic.apply_fn({"params": critic.params},
                           observations,
                           enable_dropout,
                           rngs={"dropout": rng})

  return q_vals


@jax.jit
def _update_critic(
    rng: PRNGKey,
    critic: TrainState,
    target_params: VariableDict,
    batch: FrozenDict,
    discount: float,
) -> Tuple[TrainState, MetricDict]:
  rng_1, rng_2 = jax.random.split(rng)
  target_q_vals = _critic_apply(rng_1, critic.replace(params=target_params),
                                batch["next_observations"], True)
  target_q_vals = jax.lax.stop_gradient(
      jnp.max(target_q_vals, axis=-1, keepdims=True))

  def loss_fn(params):
    q_vals = _critic_apply(rng_2, critic.replace(params=params),
                           batch["observations"], True)
    q_vals = jnp.take_along_axis(q_vals, batch["actions"], axis=-1)

    return q_learning_loss(q_vals, target_q_vals, batch["rewards"], discount,
                           batch["masks"])

  loss, grads = jax.value_and_grad(loss_fn)(critic.params)
  critic = critic.apply_gradients(grads=grads)

  return critic, {"q_loss": loss}


class DQN(DiscreteAgent):
  @staticmethod
  def get_default_config(
      updates: Optional[ConfigDictLike] = None) -> ConfigDict:
    config = ConfigDict()
    config.discount = 0.98
    config.lr = 0.001
    config.target_update_period = 10
    config.eps = 0.1

    config.mlp_args = ConfigDict()
    config.mlp_args.hidden_dim = 256
    config.mlp_args.activations = "relu-relu"
    config.mlp_args.dtype = jnp.float32
    config.mlp_args.kernel_init = kernel_default()
    config.mlp_args.bias_init = bias_default()
    config.mlp_args.intermediate_dropout = 0.1
    config.mlp_args.final_dropout = None

    if updates:
      config.update(ConfigDict(updates).copy_and_resolve_references())

    return config

  def __init__(self, config: ConfigDict, rng: PRNGKey, obs_dim: int,
               n_actions: int) -> None:
    super().__init__(config, obs_dim, n_actions)

    rng, q_init_rng = jax.random.split(rng)
    fake_obs = jnp.zeros((1, self.obs_dim))
    critic = StateDiscreteActionValue(n_actions=self.n_actions,
                                      **self.config.mlp_args.to_dict())
    critic_params = init_model(critic, q_init_rng, fake_obs)["params"]
    self._critic = TrainState.create(apply_fn=critic.apply,
                                     params=critic_params,
                                     tx=adam(self.config.lr))

    self._tgt_params = TargetParams(critic=critic_params)
    self._rng = rng

  def sample_actions(self, observations: Array) -> chex.ArrayDevice:
    """Use epsilon-greedy to sample actions."""
    self._rng, eps_rng, a_rng, dropout_rng = jax.random.split(self._rng, 4)
    shape = observations.shape[:-1] + (1,)
    eps = jax.random.uniform(eps_rng, shape)
    uniform_actions = jax.random.randint(a_rng, shape, 0, self.n_actions)
    eval_actions = _critic_apply(dropout_rng, self._critic, observations, True)
    eval_actions = jnp.argmax(eval_actions, axis=-1, keepdims=True)

    return jax.lax.select(eps > self.config.eps, eval_actions, uniform_actions)

  def eval_actions(self, observations: Array) -> chex.ArrayDevice:
    actions = _critic_apply(self._rng, self._critic, observations, False)

    return jnp.argmax(actions, axis=-1, keepdims=True)

  def update(self, batch: FrozenDict) -> MetricDict:
    self._rng, q_rng = jax.random.split(self._rng)

    self._critic, metrics = _update_critic(q_rng, self._critic,
                                           self._tgt_params.critic, batch,
                                           self.config.discount)

    if self.step % self.config.target_update_period == 0 and self.step != 0:
      self._tgt_params = TargetParams(critic=self._critic.params)

    self._step += 1

    return metrics

  def save_checkpoint(
      self,
      ckpt_dir: str,
      prefix: str = "dqn_ckpt_",
      keep: int = 1,
      overwrite: bool = False,
      keep_every_n_steps: Optional[int] = None,
  ) -> None:
    config_dict = {
        "_obs_dim": self.obs_dim,
        "_n_actions": self.n_actions,
        "_rng": jax.device_get(self._rng),
        "_step": self.step,
    }
    params_dict = {
        "critic": self._critic,
        "target_params": self._tgt_params.to_dict(),
    }

    save_checkpoint(
        ckpt_dir,
        FrozenDict({
            "config": config_dict,
            "params": params_dict
        }),
        self.step,
        prefix,
        keep,
        overwrite,
        keep_every_n_steps,
    )

  def restore_checkpoint(self,
                         ckpt_dir,
                         prefix: str = "dqn_ckpt_",
                         step: Optional[int] = None) -> None:
    state_dict = FrozenDict(restore_checkpoint(ckpt_dir, None, step, prefix))

    for k, v in state_dict["config"].items():
      setattr(self, k, jax.device_put(v) if isinstance(v, Array) else v)

    self._critic = self._critic.replace(**state_dict["params"]["critic"])
    self._tgt_params = TargetParams(**state_dict["params"]["target_params"])
