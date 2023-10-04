"""Utilities."""

from .types import Array, PRNGKey, VariableDict

from jutils import jax, np, jit

import dejax
import flax.linen as nn
from flax import struct
from functools import partial


@partial(jit, static_argnames=("axis", "repeats"))
def expand_and_repeat(a: Array, axis: int, repeats: int) -> Array:
  a = np.expand_dims(a, axis=axis)
  return np.repeat(a, repeats=repeats, axis=axis)


@jit
def soft_update(src: VariableDict, target: VariableDict, tau: float) -> VariableDict:
  return jax.tree_map(lambda s, t: s * tau + t * (1 - tau), src, target)


@struct.dataclass
class TransitionTuple:
  obs: Array
  action: Array
  reward: Array
  next_obs: Array
  done: Array


class ReplayBuffer(object):
  def __init__(self, init_transition: TransitionTuple, capacity: int = 1000000) -> None:
    self._size = capacity
    self._buffer = dejax.uniform_replay(self._size)
    self._state = self._buffer.init_fn(init_transition)

  def add(self, transition: TransitionTuple) -> None:
    self._state = self._buffer.add_fn(self._state, transition)

  def sample(self, rng: PRNGKey, size: int = 32) -> TransitionTuple:
    return self._buffer.sample_fn(self._state, rng, size)

  @property
  def capacity(self) -> int:
    return self._size
