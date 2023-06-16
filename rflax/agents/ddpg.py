"""Implementation of DDPG."""

from rflax.agents import Agent, TargetParams
from rflax.components.nets.policy import DetTanhPolicy
from rflax.components.nets.value import StateActionValue
from rflax.components.noise import add_normal_noise
from rflax.components.loss import q_learning_loss
from rflax.utils import init_model, model_apply, soft_target_update
from rflax.components.initializers import kernel_default, bias_default
from rflax.types import Array, PRNGKey, ConfigDictLike, MetricDict

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from ml_collections import ConfigDict
from optax import adam
from typing import Sequence, Optional


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
    actions = model_apply(dropout_rng, self._actor.params, self._actor.apply_fn,
                          observations, True)
    actions = jnp.where(actions > 0, actions * self.action_high,
                        -actions * self.action_low)

    return add_normal_noise(noise_rng, actions, self.config.noise_mean,
                            self.config.noise_std)

  def eval_actions(self, observations: Array) -> Array:
    actions = model_apply(self._rng, self._actor.params, self._actor.apply_fn,
                          observations, False)
    actions = jnp.where(actions > 0, actions * self.action_high,
                        -actions * self.action_low)

    return actions

  def update(self, batch: FrozenDict) -> MetricDict:
    self._rng, *rngs = jax.random.split(self._rng, 9)
    metrics = dict()

    next_actions = model_apply(
        rngs.pop(),
        self._actor.params,
        self._actor.apply_fn,
        batch["next_observations"],
        True,
    )
    tgt_qs = model_apply(
        rngs.pop(),
        self._tgt_params.critic,
        self._critic.apply_fn,
        batch["next_observations"],
        next_actions,
        True,
    )

    @jax.jit
    def critic_loss(params):
      qs = model_apply(
          rngs.pop(),
          params,
          self._critic.apply_fn,
          batch["observations"],
          batch["actions"],
          True,
      )
      loss = q_learning_loss(qs, tgt_qs, batch["rewards"], self.config.discount,
                             batch["masks"])

      return loss

    loss, grads = jax.value_and_grad(critic_loss)(self._critic.params)
    self._critic = self._critic.apply_gradients(grads=grads)
    self._tgt_params.critic = soft_target_update(self._critic.params,
                                                 self._tgt_params.critic,
                                                 self.config.tau)
    metrics["critic_loss"] = loss

    @jax.jit
    def actor_loss(params):
      actions = model_apply(rngs.pop(), params, self._actor.apply_fn,
                            batch["observations"], True)
      qs = model_apply(
          rngs.pop(),
          self._critic.params,
          self._critic.apply_fn,
          batch["observations"],
          actions,
          True,
      )
      loss = -jnp.mean(qs)

      return loss

    loss, grads = jax.value_and_grad(actor_loss)(self._actor.params)
    self._actor = self._actor.apply_gradients(grads=grads)
    self._tgt_params.actor = soft_target_update(self._actor.params,
                                                self._tgt_params.actor,
                                                self.config.tau)
    metrics["actor_loss"] = loss

    return metrics
