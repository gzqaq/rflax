"""Functions for adding noise to actions."""

from rflax.types import Array, PRNGKey

import chex
import jax
from typing import Union


@jax.jit
def add_normal_noise(
    rng: PRNGKey,
    array: Array,
    mean: Union[Array, float] = 0,
    std: Union[Array, float] = 0.1,
) -> chex.ArrayDevice:
  noise = jax.random.normal(rng, array.shape, array.dtype) * std + mean

  return array + noise
