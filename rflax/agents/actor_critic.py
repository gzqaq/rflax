"""Implementation of Actor-Critic algorithms."""

from rflax.agents import ContinuousAgent
from rflax.components.nets.policy import NormalTanhPolicy
from rflax.components.nets.value import StateValue
from rflax.components.initializers import kernel_default, bias_default
from rflax.components.loss import q_learning_loss
from rflax.utils import init_model
from rflax.types import Array, MetricDict, PRNGKey, ConfigDictLike

import chex
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from functools import partial
from ml_collections import ConfigDict
from typing import Tuple, Optional, Sequence


@partial(jax.jit, static_argnames=("enable_dropout",))
def _actor_apply(
    rng: PRNGKey, actor: TrainState, observations: Array,
    enable_dropout: bool) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
  dropout_rng, sample_rng = jax.random.split(rng)
  return actor.apply_fn(
      {
          "params": actor.params
      },
      observations,
      enable_dropout,
      rngs={
          "dropout": dropout_rng
      },
  ).sample_and_log_prob(seed=sample_rng)


@partial(jax.jit, static_argnames=("enable_dropout",))
def _critic_apply(rng: PRNGKey, critic: TrainState, observations: Array,
                  enable_dropout: bool) -> chex.ArrayDevice:
  return critic.apply_fn({"params": critic.params},
                         observations,
                         enable_dropout,
                         rngs={"dropout": rng})


@partial(jax.jit, static_argnames=("enable_dropout",))
def _log_prob_of_action(
    rng: PRNGKey,
    actor: TrainState,
    observations: Array,
    actions: Array,
    enable_dropout: bool,
) -> chex.ArrayDevice:
  return actor.apply_fn({
      "params": actor.params
  },
                        observations,
                        enable_dropout,
                        rngs={
                            "dropout": rng
                        }).log_prob(actions)


@jax.jit
def _update_actor(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: FrozenDict,
    discount: float,
) -> Tuple[TrainState, MetricDict]:
  tdtgt_rng, tddt_rng, logprob_rng = jax.random.split(rng, 3)
  td_target = (batch["rewards"] + discount * _critic_apply(
      tdtgt_rng, critic, batch["next_observations"], True) * batch["masks"])
  td_delta = td_target - _critic_apply(tddt_rng, critic, batch["observations"],
                                       True)

  def loss_fn(params):
    logprobs = _log_prob_of_action(
        logprob_rng,
        actor.replace(params=params),
        batch["observations"],
        batch["actions"],
        True,
    )
    loss = -logprobs * jax.lax.stop_gradient(td_delta)

    return jnp.mean(loss)

  loss, grads = jax.value_and_grad(loss_fn)(actor.params)
  return actor.apply_gradients(grads=grads), {"actor_loss": loss}


@jax.jit
def _update_critic(rng: PRNGKey, critic: TrainState, batch: FrozenDict,
                   discount: float) -> Tuple[TrainState, MetricDict]:
  nv_rng, v_rng = jax.random.split(rng)
  next_val = _critic_apply(nv_rng, critic, batch["next_observations"], True)

  def loss_fn(params):
    vals = _critic_apply(v_rng, critic.replace(params=params),
                         batch["observations"], True)
    return q_learning_loss(
        vals,
        jax.lax.stop_gradient(next_val),
        batch["rewards"],
        discount,
        batch["masks"],
    )

  loss, grads = jax.value_and_grad(loss_fn)(critic.params)
  return critic.apply_gradients(grads=grads), {"critic_loss": loss}


class ActorCritic(ContinuousAgent):
  @staticmethod
  def get_default_config(
      updates: Optional[ConfigDictLike] = None) -> ConfigDict:
    config = ConfigDict()
    config.discount = 0.98
    config.actor_lr = 5e-4
    config.critic_lr = 5e-3

    config.mlp_args = ConfigDict()
    config.mlp_args.hidden_dim = 2048
    config.mlp_args.activations = "relu"
    config.mlp_args.dtype = jnp.float32
    config.mlp_args.kernel_init = kernel_default()
    config.mlp_args.bias_init = bias_default()
    config.mlp_args.intermediate_dropout = 0.1
    config.mlp_args.final_dropout = None

    if updates:
      config.update(ConfigDict(updates).copy_and_resolve_references())

    return config

  def __init__(
      self,
      config: ConfigDict,
      rng: PRNGKey,
      obs_dim: int,
      action_bound: Sequence[Array],
  ) -> None:
    super().__init__(config, obs_dim, action_bound)

    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    fake_obs = jnp.zeros((1, self.obs_dim))

    actor = NormalTanhPolicy(action_dim=self.action_dim,
                             high=self.action_high,
                             low=self.action_low,
                             **self.config.mlp_args.to_dict())
    actor_params = init_model(actor, actor_rng, fake_obs)["params"]
    self._actor = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optax.adam(self.config.actor_lr),
    )

    critic = StateValue(**self.config.mlp_args.to_dict())
    critic_params = init_model(critic, critic_rng, fake_obs)["params"]
    self._critic = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=optax.adam(self.config.critic_lr),
    )

    self._rng = rng

  def sample_actions(self, observations: Array) -> chex.ArrayDevice:
    self._rng, dropout_rng = jax.random.split(self._rng)
    actions, _ = _actor_apply(dropout_rng, self._actor, observations, True)

    return actions

  def eval_actions(self, observations: Array) -> chex.ArrayDevice:
    actions, _ = _actor_apply(
        self._rng,
        self._actor,
        observations,
        False,
    )

    return actions

  def update(self, batch: FrozenDict) -> MetricDict:
    self._rng, critic_rng, actor_rng = jax.random.split(self._rng, 3)

    self._critic, metrics = _update_critic(
        critic_rng,
        self._critic,
        batch,
        self.config.discount,
    )
    self._actor, actor_info = _update_actor(actor_rng, self._actor,
                                            self._critic, batch,
                                            self.config.discount)

    metrics.update(actor_info)
    self._step += 1

    return metrics

  def save_checkpoint(
      self,
      ckpt_dir: str,
      prefix: str = "ac_ckpt_",
      keep: int = 1,
      overwrite: bool = False,
      keep_every_n_steps: Optional[int] = None,
  ) -> None:
    config_dict = {
        "_obs_dim": self.obs_dim,
        "_action_dim": self.action_dim,
        "action_high": jax.device_get(self.action_high),
        "action_low": jax.device_get(self.action_low),
        "_rng": jax.device_get(self._rng),
        "_step": self.step,
    }
    params_dict = {
        "actor": self._actor,
        "critic": self._critic,
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
                         ckpt_dir: str,
                         prefix: str = "ac_ckpt_",
                         step: Optional[int] = None) -> None:
    state_dict = restore_checkpoint(ckpt_dir, None, step, prefix)
    state_dict = FrozenDict(state_dict)

    for k, v in state_dict["config"].items():
      setattr(self, k, jax.device_put(v) if isinstance(v, Array) else v)

    self._actor = self._actor.replace(**state_dict["params"]["actor"])
    self._critic = self._critic.replace(**state_dict["params"]["critic"])
