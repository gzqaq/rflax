"""Useful utility functions."""

from rflax.types import Array

import jax.numpy as jnp


def expand_and_repeat(array: Array, axis: int, repeats: int) -> Array:
  array = jnp.expand_dims(array, axis=axis)
  return jnp.repeat(array, repeats=repeats, axis=axis)
