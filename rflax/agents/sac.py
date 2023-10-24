from ..components.blocks import MlpConfig, ScalarBlock
from ..components.loss import q_learning_loss
from ..components.nets.policy import TanhGaussianPolicy, tanh_gaussian_dist
from ..components.nets.value import ActionValueEnsemble
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

from jutils import jax, np, jit, rng_wrapper, tile_over_axis

import optax
from flax import struct
from typing import Tuple, Callable


@struct.dataclass
class SACConfig:
  n_qs: int = 2
  discount: float = 0.98
  tau: float = 0.005
  actor_lr: float = 1e-3
  critic_lr: float = 1e-3
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  alpha_multiplier: float = 1.0
  target_entropy: float = 0.0
  backup_entropy: bool = False


@struct.dataclass
class SACParams:
  actor: VariableDict
  critic: VariableDict
  target_actor: VariableDict
  target_critic: VariableDict
  log_alpha: VariableDict


@struct.dataclass
class TrainState:
  actor: OptState
  critic: OptState
  log_alpha: OptState
  step: int


def get_policy(mlp_conf: MlpConfig, action_low: Array,
               action_high: Array) -> Tuple[ApplyFunction, SamplePolicy]:
  action_dim = action_low.shape[-1]

  def policy(params: VariableDict, observation: Array) -> Tuple[Array, Array]:
    mean, log_std = get_apply_fn(TanhGaussianPolicy,
                                 action_dim,
                                 config=mlp_conf)(params, observation)
    a = np.tanh(mean)

    return np.where(a > 0, a * action_high,
                    -a * action_low), tanh_gaussian_dist(mean,
                                                         log_std).log_prob(a)

  def sample(rng: PRNGKey, params: VariableDict,
             observation: Array) -> Tuple[Array, Array]:
    mean, log_std = get_apply_fn(TanhGaussianPolicy,
                                 action_dim,
                                 config=mlp_conf)(params, observation)
    return tanh_gaussian_dist(mean, log_std).sample_and_log_prob(seed=rng)

  return jit(policy), jit(sample)


def get_critic(sac_conf: SACConfig, mlp_conf: MlpConfig) -> ApplyFunction:
  return jit(ActionValueEnsemble(sac_conf.n_qs, mlp_conf).apply)


def init_params(
    rng: PRNGKey,
    sac_conf: SACConfig,
    mlp_conf: MlpConfig,
    init_obs: Array,
    action_high: Array,
) -> SACParams:
  actor = TanhGaussianPolicy(
      action_high.shape[-1],
      sac_conf.log_std_multiplier,
      sac_conf.log_std_offset,
      mlp_conf,
      name="SAC_actor",
  )
  critic = ActionValueEnsemble(sac_conf.n_qs, mlp_conf, name="SAC_critic")
  log_alpha = ScalarBlock(0.0)

  rng, actor_params = rng_wrapper(actor.init)(rng, init_obs)
  rng, critic_params = rng_wrapper(critic.init)(rng, init_obs, action_high)
  alpha_params = log_alpha.init(rng)

  return SACParams(actor_params, critic_params, actor_params, critic_params,
                   alpha_params)


def init_train(params: SACParams, optimizer: Optimizer) -> TrainState:
  return TrainState(
      actor=optimizer.init(params.actor),
      critic=optimizer.init(params.critic),
      log_alpha=optimizer.init(params.log_alpha),
      step=0,
  )


def make_train(
    config: SACConfig,
    policy: SamplePolicy,
    critic: ApplyFunction,
    actor_opt: Optimizer,
    critic_opt: Optimizer,
    alpha_opt: Optimizer,
) -> Callable[[PRNGKey, SACParams, TrainState, TransitionTuple], Tuple[
    SACParams, TrainState, DataDict],]:
  def update_step(
      rng: PRNGKey, params: SACParams, train_state: TrainState,
      batch: TransitionTuple) -> Tuple[SACParams, TrainState, DataDict]:
    s = batch.obs
    a = batch.action
    r = batch.reward.reshape(-1, 1)
    s_ = batch.next_obs
    d = batch.done
    losses = {}

    log_alpha_fn = ScalarBlock(0.0).apply
    alpha = np.exp(log_alpha_fn(params.log_alpha)) * config.alpha_multiplier

    # actor
    def loss_fn(param):
      a, log_prob = policy(rng, param, s)
      qs = critic(params.critic, s, a).min(axis=0)

      return (alpha * log_prob - qs).mean(), (a, log_prob)

    (losses["actor"],
     (a_, log_prob)), grads = jax.value_and_grad(loss_fn,
                                                 has_aux=True)(params.actor)
    updates, new_opt_actor = actor_opt.update(grads, train_state.actor,
                                              params.actor)
    new_param_actor = optax.apply_updates(params.actor, updates)
    new_tgt_actor = soft_update(new_param_actor, params.target_actor,
                                config.tau)

    # alpha
    def loss_fn(param, log_prob):
      return -log_alpha_fn(param) * (log_prob + config.target_entropy).mean()

    losses["log_alpha"], grads = jax.value_and_grad(loss_fn)(params.log_alpha,
                                                             log_prob)
    updates, new_opt_log_alpha = alpha_opt.update(grads, train_state.log_alpha,
                                                  params.log_alpha)
    new_param_log_alpha = optax.apply_updates(params.log_alpha, updates)

    # critic
    a_, log_prob = policy(rng, params.actor, s_)
    tgt_qs = (critic(params.target_critic, s_, a_).min(axis=0) -
              alpha * log_prob * config.backup_entropy)
    tgt_qs, r, masks = jax.tree_map(lambda x: tile_over_axis(x, config.n_qs, 0),
                                    (tgt_qs, r, 1 - d))

    def loss_fn(param):
      qs = critic(param, s, a)
      return q_learning_loss(qs, tgt_qs, r, config.discount, masks)

    losses["critic"], grads = jax.value_and_grad(loss_fn)(params.critic)
    updates, new_opt_critic = critic_opt.update(grads, train_state.critic,
                                                params.critic)
    new_param_critic = optax.apply_updates(params.critic, updates)
    new_tgt_critic = soft_update(new_param_critic, params.target_critic,
                                 config.tau)

    return (
        params.replace(
            actor=new_param_actor,
            critic=new_param_critic,
            log_alpha=new_param_log_alpha,
            target_actor=new_tgt_actor,
            target_critic=new_tgt_critic,
        ),
        train_state.replace(actor=new_opt_actor,
                            critic=new_opt_critic,
                            log_alpha=new_opt_log_alpha),
        losses,
    )

  return jit(update_step)
