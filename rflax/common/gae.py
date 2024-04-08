from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as np


class GAEReturn(NamedTuple):
  advantages: jax.Array
  td_targets: jax.Array


def compute_gae_adv(
    reward: jax.Array,
    val: jax.Array,
    next_val: jax.Array,
    mask: jax.Array,
    lmbda: float | jax.Array,
    discount: float | jax.Array,
) -> GAEReturn:

  def step(
      adv: jax.Array, sample: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
  ) -> tuple[jax.Array, jax.Array]:
    mask, v, r, v_ = sample
    delta = r + mask * discount * v_ - v
    adv = delta + mask * discount * lmbda * adv

    return adv, adv

  _, adv_lst = lax.scan(
      f=step,
      init=np.zeros_like(val[0]),
      xs=(mask, val, reward, next_val),
      reverse=True,
  )
  return GAEReturn(advantages=adv_lst, td_targets=adv_lst + val)
