"""Functions for adding noise to actions."""

from rflax.types import Array, PRNGKey

import jax
from typing import Union


def add_normal_noise(
    rng: PRNGKey,
    array: Array,
    mean: Union[Array, float] = 0,
    std: Union[Array, float] = 1.0,
) -> Array:
  noise = jax.random.normal(rng, array.shape, array.dtype) + mean

  return array + noise * std
