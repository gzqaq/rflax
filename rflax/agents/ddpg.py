"""Implementation of DDPG."""

from rflax.agents import ContinuousAgent
from rflax.components.blocks import MlpConfig
from rflax.components.nets.policy import DetTanhPolicy
from rflax.components.nets.value import StateActionValue, StateActionEnsemble
from rflax.components.noise import add_normal_noise
from rflax.components.loss import q_learning_loss
from rflax.utils import init_model, soft_target_update
from rflax.types import Array, PRNGKey, MetricDict, VariableDict

import chex
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from functools import partial
from ml_collections import ConfigDict
from optax import adam
from typing import Sequence, Optional, Tuple


@partial(jax.jit, static_argnames=("enable_dropout",))
def _actor_apply(
    rng: PRNGKey,
    actor: TrainState,
    observations: Array,
    enable_dropout: bool,
    action_bound: Tuple[Array, Array],
) -> chex.ArrayDevice:
  action = actor.apply_fn({"params": actor.params},
                          observations,
                          enable_dropout,
                          rngs={"dropout": rng})
  return jnp.where(action > 0, action * action_bound[0],
                   -action * action_bound[1])


@partial(jax.jit, static_argnames=("enable_dropout",))
def _critic_apply(
    rng: PRNGKey,
    critic: TrainState,
    observations: Array,
    actions: Array,
    enable_dropout: bool,
) -> chex.ArrayDevice:
  return critic.apply_fn(
      {"params": critic.params},
      observations,
      actions,
      enable_dropout,
      rngs={"dropout": rng},
  )


@chex.chexify
@jax.jit
def _update_ddpg_critic(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_params: VariableDict,
    batch: FrozenDict,
    discount: float,
    tau: float,
    action_bound: Tuple[Array, Array],
) -> Tuple[TrainState, VariableDict, MetricDict]:
  rng_1, rng_2, rng_3 = jax.random.split(rng, 3)
  next_actions = _actor_apply(rng_1, actor, batch["next_observations"], True,
                              action_bound)
  tgt_qs = _critic_apply(
      rng_2,
      critic.replace(params=target_params),
      batch["next_observations"],
      next_actions,
      True,
  )

  def loss_fn(params):
    qs = _critic_apply(
        rng_3,
        critic.replace(params=params),
        batch["observations"],
        batch["actions"],
        True,
    )
    loss = q_learning_loss(qs, tgt_qs, batch["rewards"], discount,
                           batch["masks"])

    return loss

  loss, grads = jax.value_and_grad(loss_fn)(critic.params)

  chex.assert_tree_all_finite(grads)  # Abort if gradients are infinite

  critic = critic.apply_gradients(grads=grads)
  target_params = soft_target_update(critic.params, target_params, tau)

  return critic, target_params, {"critic_loss": loss}


@chex.chexify
@jax.jit
def _update_ddpg_actor(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_params: VariableDict,
    batch: FrozenDict,
    tau: float,
    action_bound: Tuple[Array, Array],
) -> Tuple[TrainState, VariableDict, MetricDict]:
  rng_1, rng_2 = jax.random.split(rng)

  def loss_fn(params):
    actions = _actor_apply(
        rng_1,
        actor.replace(params=params),
        batch["observations"],
        True,
        action_bound,
    )
    qs = _critic_apply(rng_2, critic, batch["observations"], actions, True)
    loss = -jnp.mean(qs)

    return loss

  loss, grads = jax.value_and_grad(loss_fn)(actor.params)

  chex.assert_tree_all_finite(grads)  # Abort if gradients are infinite

  actor = actor.apply_gradients(grads=grads)
  target_params = soft_target_update(actor.params, target_params, tau)

  return actor, target_params, {"actor_loss": loss}


@chex.dataclass(frozen=True)
class DDPGConfig:
  discount: float = 0.98
  tau: float = 0.005
  actor_lr: float = 0.001
  critic_lr: float = 0.001
  act_noise: float = 0.1
  mlp_args: MlpConfig = MlpConfig()

@chex.dataclass(frozen=True)
class TargetParams:
  actor: VariableDict
  critic: VariableDict

class DDPG(ContinuousAgent):
  @staticmethod
  def default_config() -> DDPGConfig:
    return DDPGConfig()

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
                          config=self.config.mlp_args)
    actor_params = init_model(actor, actor_rng, fake_obs)["params"]
    self._actor = TrainState.create(apply_fn=actor.apply,
                                    params=actor_params,
                                    tx=adam(self.config.actor_lr))

    critic = StateActionValue(self.config.mlp_args)
    critic_params = init_model(critic, critic_rng, fake_obs,
                               fake_actions)["params"]
    self._critic = TrainState.create(apply_fn=critic.apply,
                                     params=critic_params,
                                     tx=adam(self.config.critic_lr))

    self._tgt_params = TargetParams(actor=actor_params, critic=critic_params)
    self._rng = rng

  def sample_actions(self, observations: Array) -> chex.ArrayDevice:
    self._rng, noise_rng, dropout_rng = jax.random.split(self._rng, 3)
    actions = _actor_apply(
        dropout_rng,
        self._actor,
        observations,
        True,
        (self.action_high, self.action_low),
    )

    return add_normal_noise(noise_rng, actions, 0,
                            self.config.act_noise)

  def eval_actions(self, observations: Array) -> chex.ArrayDevice:
    actions = _actor_apply(
        self._rng,
        self._actor,
        observations,
        False,
        (self.action_high, self.action_low),
    )

    return actions

  def update(self, batch: FrozenDict) -> MetricDict:
    self._rng, critic_rng, actor_rng = jax.random.split(self._rng, 3)

    self._critic, critic_target_params, metrics = _update_ddpg_critic(
        critic_rng,
        self._actor.replace(params=self._tgt_params.actor),
        self._critic,
        self._tgt_params.critic,
        batch,
        self.config.discount,
        self.config.tau,
        (self.action_high, self.action_low),
    )
    self._actor, actor_target_params, actor_info = _update_ddpg_actor(
        actor_rng,
        self._actor,
        self._critic,
        self._tgt_params.actor,
        batch,
        self.config.tau,
        (self.action_high, self.action_low),
    )
    chex.block_until_chexify_assertions_complete()

    metrics.update(actor_info)
    self._tgt_params = self._tgt_params.replace(actor=actor_target_params,
                                    critic=critic_target_params)
    self._step += 1

    return metrics

  def save_checkpoint(
      self,
      ckpt_dir: str,
      prefix: str = "ddpg_ckpt_",
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
        "target_params": dict(self._tgt_params.items()),
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
                         prefix: str = "ddpg_ckpt_",
                         step: Optional[int] = None) -> None:
    state_dict = restore_checkpoint(ckpt_dir, None, step, prefix)
    state_dict = FrozenDict(state_dict)

    for k, v in state_dict["config"].items():
      setattr(self, k, jax.device_put(v) if isinstance(v, Array) else v)

    self._actor = self._actor.replace(**state_dict["params"]["actor"])
    self._critic = self._critic.replace(**state_dict["params"]["critic"])
    self._tgt_params = TargetParams(**state_dict["params"]["target_params"])


@chex.chexify
@jax.jit
def _update_td3_critic(
    rng,
    actor,
    critic,
    target_param,
    batch,
    discount,
    tau,
    noise_std,
    clip_noise,
    action_bound,
):
  rewards = jnp.stack([batch["rewards"].reshape(-1, 1)] * 2)
  masks = jnp.stack([batch["masks"].reshape(-1, 1)] * 2)
  noise_rng, rng_1, rng_2, rng_3 = jax.random.split(rng, 4)
  next_actions = _actor_apply(rng_1, actor, batch["next_observations"], True,
                              action_bound)
  noise = jax.random.normal(noise_rng, next_actions.shape) * noise_std
  next_actions = jnp.clip(
      next_actions + jnp.clip(noise, -clip_noise, clip_noise),
      action_bound[-1],
      action_bound[0],
  )
  tgt_qs = _critic_apply(
      rng_2,
      critic.replace(params=target_param),
      batch["next_observations"],
      next_actions,
      True,
  ).min(axis=0)

  def loss_fn(params):
    qs = _critic_apply(
        rng_3,
        critic.replace(params=params),
        batch["observations"],
        batch["actions"],
        True,
    )
    loss = q_learning_loss(qs, jnp.stack([tgt_qs] * 2), rewards, discount,
                           masks)

    return loss

  loss, grads = jax.value_and_grad(loss_fn)(critic.params)

  chex.assert_tree_all_finite(grads)  # Abort if gradients are infinite

  critic = critic.apply_gradients(grads=grads)
  target_params = soft_target_update(critic.params, target_param, tau)

  return critic, target_params, {"critic_loss": loss}


@chex.chexify
@jax.jit
def _update_td3_actor(rng, actor, critic, target_param, batch, tau,
                      action_bound):
  rng_1, rng_2 = jax.random.split(rng)

  def loss_fn(params):
    actions = _actor_apply(
        rng_1,
        actor.replace(params=params),
        batch["observations"],
        True,
        action_bound,
    )
    qs = _critic_apply(rng_2, critic, batch["observations"], actions, True)[0]
    loss = -jnp.mean(qs)

    return loss

  loss, grads = jax.value_and_grad(loss_fn)(actor.params)

  chex.assert_tree_all_finite(grads)  # Abort if gradients are infinite

  actor = actor.apply_gradients(grads=grads)
  target_params = soft_target_update(actor.params, target_param, tau)

  return actor, target_params, {"actor_loss": loss}


@chex.dataclass(frozen=True)
class TD3Config:
  num_qs: int = 2
  discount: float = 0.98
  tau: float = 0.005
  actor_lr: float = 0.001
  critic_lr: float = 0.001
  act_noise: float = 0.1
  target_noise: float = 0.2
  clip_noise: float = 0.5
  policy_delay: int = 2
  mlp_args: MlpConfig = MlpConfig()

class TD3(ContinuousAgent):
  @staticmethod
  def default_config() -> TD3Config:
    return TD3Config()

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
                          config=self.config.mlp_args)
    actor_params = init_model(actor, actor_rng, fake_obs)["params"]
    self._actor = TrainState.create(apply_fn=actor.apply,
                                    params=actor_params,
                                    tx=adam(self.config.actor_lr))

    critic = StateActionEnsemble(num_qs=self.config.num_qs,
                                 config=self.config.mlp_args)
    critic_params = init_model(critic, critic_rng, fake_obs,
                               fake_actions)["params"]
    self._critic = TrainState.create(apply_fn=critic.apply,
                                     params=critic_params,
                                     tx=adam(self.config.critic_lr))

    self._tgt_params = TargetParams(actor=actor_params, critic=critic_params)
    self._rng = rng

  def sample_actions(self, observations: Array) -> chex.ArrayDevice:
    self._rng, noise_rng, dropout_rng = jax.random.split(self._rng, 3)
    actions = _actor_apply(
        dropout_rng,
        self._actor,
        observations,
        True,
        (self.action_high, self.action_low),
    )

    return add_normal_noise(noise_rng, actions, 0, self.config.act_noise)

  def eval_actions(self, observations: Array) -> chex.ArrayDevice:
    actions = _actor_apply(
        self._rng,
        self._actor,
        observations,
        False,
        (self.action_high, self.action_low),
    )

    return actions

  def update(self, batch: FrozenDict) -> MetricDict:
    self._rng, critic_rng, actor_rng = jax.random.split(self._rng, 3)

    self._critic, critic_tgt_param, metrics = _update_td3_critic(
        critic_rng,
        self._actor.replace(params=self._tgt_params.actor),
        self._critic,
        self._tgt_params.critic,
        batch,
        self.config.discount,
        self.config.tau,
        self.config.target_noise,
        self.config.clip_noise,
        (self.action_high, self.action_low),
    )

    if (self.step + 1) % self.config.policy_delay == 0:
      self._actor, actor_tgt_param, actor_info = _update_td3_actor(
          actor_rng,
          self._actor,
          self._critic,
          self._tgt_params.actor,
          batch,
          self.config.tau,
          (self.action_high, self.action_low),
      )
      metrics.update(actor_info)
      self._tgt_params = self._tgt_params.replace(actor=actor_tgt_param,
                                      critic=critic_tgt_param)
    else:
      self._tgt_params = self._tgt_params.replace(critic=critic_tgt_param)

    chex.block_until_chexify_assertions_complete()
    self._step += 1
    return metrics

  def save_checkpoint(
      self,
      ckpt_dir: str,
      prefix: str = "td3_ckpt_",
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
        "target_params": dict(self._tgt_params.items()),
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
                         prefix: str = "td3_ckpt_",
                         step: Optional[int] = None) -> None:
    state_dict = restore_checkpoint(ckpt_dir, None, step, prefix)
    state_dict = FrozenDict(state_dict)

    for k, v in state_dict["config"].items():
      setattr(self, k, jax.device_put(v) if isinstance(v, Array) else v)

    self._actor = self._actor.replace(**state_dict["params"]["actor"])
    self._critic = self._critic.replace(**state_dict["params"]["critic"])
    self._tgt_params = TargetParams(**state_dict["params"]["target_params"])
