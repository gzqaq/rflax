"""Conservative SAC."""

from .sac import (
    SACConfig,
    SACParams,
    TrainState,
    get_policy,
    get_critic,
    init_params,
    init_train,
)
from ..components.blocks import MlpConfig, ScalarBlock
from ..components.loss import q_learning_loss
from ..types import (
    Array,
    PRNGKey,
    SamplePolicy,
    VariableDict,
    DataDict,
    ApplyFunction,
    Optimizer,
)
from ..utils import TransitionTuple, soft_update

from jutils import jax, random, np, jit, rng_wrapper, split_rng, tile_over_axis

import optax
from flax import struct
from typing import Tuple, Callable


@struct.dataclass
class CQLConfig:
  n_actions: int = 10
  target_action_gap: float = 1.0
  temp: float = 1.0
  min_q_weight: float = 5.0
  clip_diff_min: float = -np.inf
  clip_diff_max: float = np.inf


def make_train(
    sac_conf: SACConfig,
    cql_conf: CQLConfig,
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
    alpha = np.exp(log_alpha_fn(params.log_alpha)) * sac_conf.alpha_multiplier

    # actor
    rng, _rng = split_rng(rng)

    def loss_fn(param):
      a, log_prob = policy(_rng, param, s)
      qs = critic(params.critic, s, a).min(axis=0)

      return (alpha * log_prob.reshape(qs.shape) - qs).mean(), (a, log_prob)

    (losses["actor"],
     (a_, log_prob)), grads = jax.value_and_grad(loss_fn,
                                                 has_aux=True)(params.actor)
    updates, new_opt_actor = actor_opt.update(grads, train_state.actor,
                                              params.actor)
    new_param_actor = optax.apply_updates(params.actor, updates)
    new_tgt_actor = soft_update(new_param_actor, params.target_actor,
                                sac_conf.tau)

    # alpha
    def loss_fn(param, log_prob):
      return -log_alpha_fn(param) * (log_prob + sac_conf.target_entropy).mean()

    losses["log_alpha"], grads = jax.value_and_grad(loss_fn)(params.log_alpha,
                                                             log_prob)
    updates, new_opt_log_alpha = alpha_opt.update(grads, train_state.log_alpha,
                                                  params.log_alpha)
    new_param_log_alpha = optax.apply_updates(params.log_alpha, updates)

    # critic
    rng, (a_, log_prob) = rng_wrapper(policy)(rng, params.actor, s_)
    tgt_qs = (critic(params.target_critic, s_, a_).min(axis=0) -
              alpha * log_prob.reshape(-1, 1) * sac_conf.backup_entropy)
    tgt_qs, r, masks = jax.tree_map(
        lambda x: tile_over_axis(x, sac_conf.n_qs, 0), (tgt_qs, r, 1 - d))
    ## CQL
    tiled_s = tile_over_axis(s, cql_conf.n_actions, 0)
    tiled_s_ = tile_over_axis(s_, cql_conf.n_actions, 0)
    random_a = random.uniform(rng, (cql_conf.n_actions, *a.shape),
                              minval=-1.0,
                              maxval=1.0)
    rng, (curr_a, curr_log_prob) = rng_wrapper(policy)(rng, params.actor,
                                                       tiled_s)
    rng, (next_a, next_log_prob) = rng_wrapper(policy)(rng, params.actor,
                                                       tiled_s_)

    def loss_fn(param):
      random_cqs = critic(param, tiled_s, random_a)
      curr_cqs = critic(param, tiled_s, curr_a)
      next_cqs = critic(param, tiled_s_, next_a)
      qs = critic(param, s, a)

      # Importance sampling
      random_density = np.log(0.5) * a.shape[-1]
      cqs = np.concatenate(
          [
              random_cqs - random_density,
              curr_cqs - curr_log_prob[..., None],
              next_cqs - next_log_prob[..., None],
          ],
          axis=1,
      )

      ood_qs = (jax.scipy.special.logsumexp(cqs / cql_conf.temp, axis=1) *
                cql_conf.temp)
      q_diff = np.clip(ood_qs - qs, cql_conf.clip_diff_min,
                       cql_conf.clip_diff_max).sum()
      q_loss = q_learning_loss(qs, tgt_qs, r, sac_conf.discount, masks)
      cql_loss = q_diff * cql_conf.min_q_weight
      return q_loss + cql_loss, (q_loss, cql_loss)

    (
        losses["critic"],
        (losses["q_loss"], losses["cql_loss"]),
    ), grads = jax.value_and_grad(loss_fn, has_aux=True)(params.critic)
    updates, new_opt_critic = critic_opt.update(grads, train_state.critic,
                                                params.critic)
    new_param_critic = optax.apply_updates(params.critic, updates)
    new_tgt_critic = soft_update(new_param_critic, params.target_critic,
                                 sac_conf.tau)

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
