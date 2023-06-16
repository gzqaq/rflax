"""Implementation of DDPG."""

from rflax.agents import Agent, TargetParams
from rflax.components.nets.policy import DetTanhPolicy
from rflax.components.nets.value import StateActionValue
from rflax.components.noise import add_normal_noise
from rflax.components.loss import q_learning_loss
from rflax.utils import init_model, model_apply, soft_target_update
from rflax.components.initializers import kernel_default, bias_default
from rflax.types import Array, PRNGKey, ConfigDictLike, MetricDict, VariableDict

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from functools import partial
from ml_collections import ConfigDict
from optax import adam
from typing import Sequence, Optional, Tuple


class DDPG(Agent):
  @staticmethod
  def get_default_config(
      updates: Optional[ConfigDictLike] = None) -> ConfigDict:
    config = ConfigDict()
    config.discount = 0.99
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.tau = 0.005
    config.noise_mean = 0.0
    config.noise_std = 1.0

    config.mlp_args = ConfigDict()
    config.mlp_args.hidden_dim = 2048
    config.mlp_args.activations = ("relu",)
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
    fake_actions = jnp.zeros((1, self.action_dim))

    actor = DetTanhPolicy(action_dim=self.action_dim,
                          **self.config.mlp_args.to_dict())
    actor_params = init_model(actor, actor_rng, fake_obs)["params"]
    self._actor = TrainState.create(apply_fn=actor.apply,
                                    params=actor_params,
                                    tx=adam(self.config.actor_lr))

    critic = StateActionValue(**self.config.mlp_args.to_dict())
    critic_params = init_model(critic, critic_rng, fake_obs,
                               fake_actions)["params"]
    self._critic = TrainState.create(apply_fn=critic.apply,
                                     params=critic_params,
                                     tx=adam(self.config.critic_lr))

    self._tgt_params = TargetParams(actor=actor_params, critic=critic_params)
    self._rng = rng

  def sample_actions(self, observations: Array) -> Array:
    self._rng, noise_rng, dropout_rng = jax.random.split(self._rng, 3)
    actions = self._actor_apply(dropout_rng, self._actor.params, observations,
                                True)
    actions = jnp.where(actions > 0, actions * self.action_high,
                        -actions * self.action_low)

    return add_normal_noise(noise_rng, actions, self.config.noise_mean,
                            self.config.noise_std)

  def eval_actions(self, observations: Array) -> Array:
    actions = self._actor_apply(self._rng, self._actor.params, observations,
                                False)
    actions = jnp.where(actions > 0, actions * self.action_high,
                        -actions * self.action_low)

    return actions

  @partial(jax.jit, static_argnames=("self", "enable_dropout"))
  def _actor_apply(
      self,
      rng: PRNGKey,
      params: VariableDict,
      observations: Array,
      enable_dropout: bool,
  ) -> Array:
    return self._actor.apply_fn({"params": params},
                                observations,
                                enable_dropout,
                                rngs={"dropout": rng})

  @partial(jax.jit, static_argnames=("self", "enable_dropout"))
  def _critic_apply(
      self,
      rng: PRNGKey,
      params: VariableDict,
      observations: Array,
      actions: Array,
      enable_dropout: bool,
  ) -> Array:
    return self._critic.apply_fn(
        {"params": params},
        observations,
        actions,
        enable_dropout,
        rngs={"dropout": rng},
    )

  @partial(jax.jit, static_argnames=("self",))
  def _update_critic(
      self,
      rng: PRNGKey,
      critic: TrainState,
      target_params: VariableDict,
      batch: FrozenDict,
  ) -> Tuple[TrainState, VariableDict, MetricDict]:
    rng_1, rng_2, rng_3 = jax.random.split(rng, 3)
    next_actions = self._actor_apply(rng_1, self._tgt_params.actor,
                                     batch["next_observations"], True)
    tgt_qs = self._critic_apply(rng_2, target_params,
                                batch["next_observations"], next_actions, True)

    @jax.jit
    def loss_fn(params):
      qs = self._critic_apply(rng_3, params, batch["observations"],
                              batch["actions"], True)
      loss = q_learning_loss(qs, tgt_qs, batch["rewards"], self.config.discount,
                             batch["masks"])

      return loss

    loss, grads = jax.value_and_grad(loss_fn)(critic.params)
    critic = critic.apply_gradients(grads=grads)
    target_params = soft_target_update(critic.params, target_params,
                                       self.config.tau)

    return critic, target_params, {"critic_loss": loss}

  @partial(jax.jit, static_argnames=("self",))
  def _update_actor(
      self,
      rng: PRNGKey,
      actor: TrainState,
      target_params: VariableDict,
      batch: FrozenDict,
  ) -> Tuple[TrainState, VariableDict, MetricDict]:
    rng_1, rng_2 = jax.random.split(rng)

    @jax.jit
    def loss_fn(params):
      actions = self._actor_apply(rng_1, params, batch["observations"], True)
      qs = self._critic_apply(rng_2, self._critic.params, batch["observations"],
                              actions, True)
      loss = -jnp.mean(qs)

      return loss

    loss, grads = jax.value_and_grad(loss_fn)(actor.params)
    actor = actor.apply_gradients(grads=grads)
    target_params = soft_target_update(actor.params, target_params,
                                       self.config.tau)

    return actor, target_params, {"actor_loss": loss}

  def update(self, batch: FrozenDict) -> MetricDict:
    self._rng, critic_rng, actor_rng = jax.random.split(self._rng, 3)

    self._critic, self._tgt_params.critic, metrics = self._update_critic(
        critic_rng, self._critic, self._tgt_params.critic, batch)
    self._actor, self._tgt_params.actor, actor_info = self._update_actor(
        actor_rng, self._actor, self._tgt_params.actor, batch)

    metrics.update(actor_info)
    return metrics
