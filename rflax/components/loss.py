"""Loss functions."""

from rflax.types import Array

import jax
import jax.numpy as jnp


@jax.jit
def q_learning_loss(q_vals: Array, target_q_vals: Array, rewards: Array,
                    discount: float, masks: Array) -> Array:
  td_target = rewards + discount * (1 - masks) * target_q_vals

  return jnp.mean(jnp.power(q_vals - td_target, 2))
