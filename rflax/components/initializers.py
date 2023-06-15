"""Initializers for rflax."""

from rflax.types import Initializer

from flax.linen.initializers import xavier_uniform, normal


def kernel_default() -> Initializer:
  return xavier_uniform()


def bias_default() -> Initializer:
  return normal(stddev=1e-6)
