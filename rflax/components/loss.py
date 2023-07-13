"""Loss functions."""

from rflax.types import Array

import chex
import jax
import jax.numpy as jnp


@jax.jit
def q_learning_loss(q_vals: Array, target_q_vals: Array, rewards: Array,
                    discount: float, masks: Array) -> chex.ArrayDevice:
  rewards = jnp.reshape(rewards, q_vals.shape)
  masks = jnp.reshape(masks, q_vals.shape)
  td_target = rewards + discount * masks * target_q_vals

  return jnp.mean(jnp.power(q_vals - td_target, 2))
