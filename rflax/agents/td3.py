from ..components.blocks import MlpConfig
from ..components.loss import q_learning_loss
from ..components.nets.policy import DetTanhPolicy
from ..components.nets.value import ActionValueEnsemble
from ..components.noise import add_normal_noise
from ..types import (
    Array,
    PRNGKey,
    SamplePolicy,
    VariableDict,
    DataDict,
    ApplyFunction,
    OptState,
    Optimizer,
)
from ..utils import get_apply_fn, TransitionTuple, soft_update

from jutils import jax, lax, np, random, jit, rng_wrapper, tile_over_axis

import optax
from flax import struct
from typing import Tuple, Callable


@struct.dataclass
class TD3Config:
  n_qs: int = 2
  discount: float = 0.98
  tau: float = 0.005
  actor_lr: float = 1e-3
  critic_lr: float = 1e-3
  act_noise: float = 0.1
  target_noise: float = 0.2
  clip_noise: float = 0.5
  policy_delay: int = 2


@struct.dataclass
class TD3Params:
  actor: VariableDict
  critic: VariableDict
  target_actor: VariableDict
  target_critic: VariableDict


@struct.dataclass
class TrainState:
  actor: OptState
  critic: OptState
  step: int


def get_policy(
    td3_conf: TD3Config,
    mlp_conf: MlpConfig,
    action_low: Array,
    action_high: Array,
) -> Tuple[ApplyFunction, SamplePolicy]:
  action_dim = action_low.shape[-1]

  def policy(params: VariableDict, observation: Array) -> Array:
    a = get_apply_fn(DetTanhPolicy, action_dim, mlp_conf)(params, observation)
    return np.where(a > 0, a * action_high, -a * action_low)

  def sample_policy(rng: PRNGKey, params: VariableDict,
                    observation: Array) -> Array:
    a = policy(params, observation)
    return add_normal_noise(rng, a, 0,
                            td3_conf.act_noise).clip(action_low, action_high)

  return jit(policy), jit(sample_policy)


def get_critic(td3_conf: TD3Config, mlp_conf: MlpConfig) -> ApplyFunction:
  return jit(ActionValueEnsemble(td3_conf.n_qs, mlp_conf).apply)


def init_params(
    rng: PRNGKey,
    td3_conf: TD3Config,
    mlp_conf: MlpConfig,
    init_obs: Array,
    action_high: Array,
) -> TD3Params:
  actor = DetTanhPolicy(action_high.shape[-1], mlp_conf, name="TD3_actor")
  critic = ActionValueEnsemble(td3_conf.n_qs, mlp_conf, name="TD3_critic")

  rng, (init_a, actor_params) = rng_wrapper(actor.init_with_output)(rng,
                                                                    init_obs)
  critic_params = critic.init(rng, init_obs, init_a)

  return TD3Params(actor_params, critic_params, actor_params, critic_params)


def init_train(state: TD3Params, optimizer: Optimizer) -> TrainState:
  return TrainState(
      actor=optimizer.init(state.actor),
      critic=optimizer.init(state.critic),
      step=0,
  )


def make_train(
    config: TD3Config,
    policy: ApplyFunction,
    critic: ApplyFunction,
    actor_opt: Optimizer,
    critic_opt: Optimizer,
    action_low: Array,
    action_high: Array,
) -> Callable[[PRNGKey, TD3Params, TrainState, TransitionTuple], Tuple[
    TD3Params, TrainState, DataDict],]:
  def update_step(
      rng: PRNGKey,
      params: TD3Params,
      train_state: TrainState,
      batch: TransitionTuple,
  ) -> Tuple[TD3Params, TrainState, DataDict]:
    def update_critic(rng: PRNGKey, state: TD3Params, opt_state: OptState):
      r = tile_over_axis(batch.reward.reshape(-1, 1), config.n_qs, 0)
      masks = tile_over_axis(1 - batch.done.reshape(-1, 1), config.n_qs, 0)

      a_ = policy(state.target_actor, batch.next_obs)
      noise = random.normal(rng, a_.shape) * config.target_noise
      a_ = np.clip(
          a_ + np.clip(noise, -config.clip_noise, config.clip_noise),
          action_low,
          action_high,
      )

      tgt_qs = critic(state.target_critic, batch.next_obs,
                      a_).min(axis=0)
      tgt_qs = tile_over_axis(tgt_qs, config.n_qs, 0)

      def loss_fn(params):
        qs = critic(params, batch.obs, batch.action)
        return q_learning_loss(qs, tgt_qs, r, config.discount, masks)

      loss, grads = jax.value_and_grad(loss_fn)(state.critic)
      updates, opt_state = critic_opt.update(grads, opt_state,
                                             state.critic)
      new_params = optax.apply_updates(state.critic, updates)
      new_tgt_params = soft_update(new_params, state.target_critic,
                                   config.tau)

      return (
          state.replace(critic=new_params,
                        target_critic=new_tgt_params),
          opt_state,
          loss,
      )

    def update_actor(state: TD3Params, opt_state: OptState):
      def loss_fn(params):
        a = policy(params, batch.obs)
        qs = critic(state.critic, batch.obs, a)[0]
        loss = -np.sum(qs)

        return loss

      loss, grads = jax.value_and_grad(loss_fn)(state.actor)
      updates, opt_state = actor_opt.update(grads, opt_state,
                                            state.actor)
      new_params = optax.apply_updates(state.actor, updates)
      new_tgt_params = soft_update(new_params, state.target_actor,
                                   config.tau)

      return (
          state.replace(actor=new_params,
                        target_actor=new_tgt_params),
          opt_state,
          loss,
      )

    params, critic_state, critic_loss = update_critic(
        rng, params, train_state.critic)

    def delay(params, train_state):
      train_state = train_state.replace(critic=critic_state,
                                        step=train_state.step + 1)
      metrics = {
          "actor_loss": np.empty_like(critic_loss),
          "critic_loss": critic_loss,
      }
      return params, train_state, metrics

    def no_delay(params, train_state):
      params, actor_state, actor_loss = update_actor(params,
                                                        train_state.actor)
      train_state = train_state.replace(actor=actor_state,
                                        critic=critic_state,
                                        step=train_state.step + 1)
      metrics = {"actor_loss": actor_loss, "critic_loss": critic_loss}
      return params, train_state, metrics

    return lax.cond(
        (train_state.step + 1) % config.policy_delay == 0,
        no_delay,
        delay,
        params,
        train_state,
    )

  return jit(update_step)
