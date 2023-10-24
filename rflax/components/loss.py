"""Loss functions."""

from rflax.types import Array

from jutils import jax, np, sum_of_square

import chex


@jax.jit
def q_learning_loss(q_vals: Array, target_q_vals: Array, rewards: Array,
                    discount: float, masks: Array) -> chex.ArrayDevice:
  rewards = np.reshape(rewards, q_vals.shape)
  masks = np.reshape(masks, q_vals.shape)
  td_target = rewards + discount * masks * target_q_vals

  return sum_of_square(q_vals - td_target)
