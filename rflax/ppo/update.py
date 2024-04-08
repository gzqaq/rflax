from collections.abc import Callable
from typing import NamedTuple, TypeAlias

import chex as cx
import jax
import jax.lax as lax
import jax.numpy as np
import jax.random as random
from flax.training.train_state import TrainState

from rflax.common.dist import TanhNormal
from rflax.common.gae import compute_gae_adv
from rflax.common.utils import construct_minibatches_and_shuffle
from rflax.type_hints import Variable

AgentApplyFn: TypeAlias = Callable[[Variable, jax.Array], tuple[TanhNormal,
                                                                jax.Array]]


class RolloutSamples(NamedTuple):
  s: jax.Array
  a: jax.Array
  log_prob: jax.Array
  val: jax.Array
  s_: jax.Array
  r: jax.Array
  d: jax.Array


class Minibatch(NamedTuple):
  s: jax.Array
  a: jax.Array
  log_prob: jax.Array
  val: jax.Array
  adv: jax.Array
  tgt: jax.Array


class TrainMetrics(NamedTuple):
  policy_loss: jax.Array
  value_loss: jax.Array
  entropy: jax.Array
  old_approx_kl: jax.Array
  approx_kl: jax.Array
  clip_frac: jax.Array


GradFn: TypeAlias = Callable[[Variable, AgentApplyFn, Minibatch],
                             tuple[tuple[jax.Array, TrainMetrics], Variable]]


def loss_and_grad(clip_eps: float, clip_vf: float | None, vf_coef: float,
                  ent_coef: float) -> GradFn:

  def loss_fn(params: Variable, apply_fn: AgentApplyFn,
              mb: Minibatch) -> tuple[jax.Array, TrainMetrics]:
    pi, val = apply_fn(params, mb.s)
    log_prob = pi.log_prob(mb.a)

    cx.assert_equal_shape((log_prob, val, mb.log_prob, mb.val, mb.adv, mb.tgt))

    # value loss
    v_loss = np.square(val - mb.tgt)
    if clip_vf is None:
      v_loss = 0.5 * v_loss.mean()
    else:
      v_clipped = mb.val + np.clip(val - mb.val, -clip_vf, clip_vf)
      loss_clipped = np.square(v_clipped - mb.tgt)
      v_loss = 0.5 * np.maximum(v_loss, loss_clipped).mean()

    # policy loss
    log_ratio = log_prob - mb.log_prob
    ratio = np.exp(log_ratio)

    adv = (mb.adv - mb.adv.mean()) / (mb.adv.std() + np.finfo(mb.adv.dtype).eps)
    a_loss = ratio * adv
    a_loss_clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    a_loss = -np.minimum(a_loss, a_loss_clipped).mean()

    entropy = pi.entropy().mean()

    # metrics
    old_approx_kl = -log_ratio.mean()
    approx_kl = (ratio - 1 - log_ratio).mean()
    clip_frac = np.mean(np.abs(ratio - 1) > clip_eps)

    loss = a_loss + vf_coef * v_loss - ent_coef * entropy

    return loss, TrainMetrics(a_loss, v_loss, entropy, old_approx_kl, approx_kl,
                              clip_frac)

  return jax.value_and_grad(loss_fn, has_aux=True)


class EpochCarry(NamedTuple):
  key: cx.PRNGKey
  agent: TrainState
  b_s: jax.Array
  b_a: jax.Array
  b_log_prob: jax.Array
  b_val: jax.Array
  b_adv: jax.Array
  b_tgt: jax.Array


EpochFn: TypeAlias = Callable[
    [
        cx.PRNGKey,
        TrainState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    tuple[TrainState, TrainMetrics],
]


def epoch_iterations(
    rollout_steps: int,
    n_epochs: int,
    n_minibatches: int,
    clip_eps: float,
    clip_vf: float | None,
    vf_coef: float,
    ent_coef: float,
) -> EpochFn:
  mb_size = rollout_steps // n_minibatches

  def epoch_loop(
      key: cx.PRNGKey,
      agent: TrainState,
      b_s: jax.Array,
      b_a: jax.Array,
      b_log_prob: jax.Array,
      b_val: jax.Array,
      b_adv: jax.Array,
      b_tgt: jax.Array,
  ) -> tuple[TrainState, TrainMetrics]:

    def run(carry: EpochCarry, _: None) -> tuple[EpochCarry, TrainMetrics]:

      def mb_run(agent: TrainState,
                 mb: Minibatch) -> tuple[TrainState, TrainMetrics]:
        cx.assert_tree_shape_prefix(mb, (mb_size,))

        (_, metrics), grads = loss_and_grad(
            clip_eps=clip_eps,
            clip_vf=clip_vf,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
        )(agent.params, agent.apply_fn, mb)

        return agent.apply_gradients(grads=grads), metrics

      key, agent, b_s, b_a, b_log_prob, b_val, b_adv, b_tgt = carry
      key, sk = random.split(key)
      mb_lst = construct_minibatches_and_shuffle(
          sk,
          Minibatch(
              s=b_s,
              a=b_a,
              log_prob=b_log_prob,
              val=b_val,
              adv=b_adv,
              tgt=b_tgt),
          n_minibatches,
          mb_size,
      )
      agent, metrics = lax.scan(f=mb_run, init=agent, xs=mb_lst)

      return (
          EpochCarry(key, agent, b_s, b_a, b_log_prob, b_val, b_adv, b_tgt),
          metrics,
      )

    cx.assert_tree_shape_prefix((b_s, b_a, b_log_prob, b_val, b_adv, b_tgt),
                                (rollout_steps,))
    carry = EpochCarry(key, agent, b_s, b_a, b_log_prob, b_val, b_adv, b_tgt)
    (_, agent, *_), metrics = lax.scan(
        f=run,  # type: ignore[arg-type]
        init=carry,  # type: ignore[arg-type]
        xs=None,
        length=n_epochs,
    )

    return agent, metrics

  return epoch_loop


UpdateFn: TypeAlias = Callable[[cx.PRNGKey, TrainState, RolloutSamples],
                               tuple[TrainState, TrainMetrics]]


def update_on_samples(
    rollout_steps: int,
    n_epochs: int,
    n_minibatches: int,
    clip_eps: float,
    clip_vf: float | None,
    vf_coef: float,
    ent_coef: float,
    discount: float,
    gae_lambda: float,
) -> UpdateFn:

  def fn(key: cx.PRNGKey, agent: TrainState,
         samples: RolloutSamples) -> tuple[TrainState, TrainMetrics]:
    _, v_ = agent.apply_fn(agent.params, samples.s_)
    b_adv, b_tgt = compute_gae_adv(
        samples.r,
        samples.val,
        v_,
        1 - samples.d.astype(v_.dtype),
        gae_lambda,
        discount,
    )

    return epoch_iterations(
        rollout_steps=rollout_steps,
        n_epochs=n_epochs,
        n_minibatches=n_minibatches,
        clip_eps=clip_eps,
        clip_vf=clip_vf,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
    )(key, agent, samples.s, samples.a, samples.log_prob, samples.val, b_adv,
      b_tgt)

  return fn
