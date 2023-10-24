"""Utilities."""

from .types import Array, PRNGKey, VariableDict

from jutils import jax, np, jit

import chex
import dejax
from typing import Callable, Any


def get_apply_fn(nn_cls, *args, **kwargs) -> Callable[[VariableDict, Any], Any]:
  return jit(nn_cls(*args, **kwargs).apply)


@jit
def soft_update(src: VariableDict, target: VariableDict,
                tau: float) -> VariableDict:
  return jax.tree_map(lambda s, t: s * tau + t * (1 - tau), src, target)


@chex.dataclass(frozen=True)
class TransitionTuple:
  obs: Array
  action: Array
  reward: Array
  next_obs: Array
  done: Array

  @classmethod
  def new(cls, obs: Array, action: Array, reward: Array, next_obs: Array,
          done: Array):
    return cls(obs=obs,
               action=action,
               reward=reward,
               next_obs=next_obs,
               done=done)

  @classmethod
  def dummy(cls, obs: Array, action: Array):
    return cls.new(obs, action, np.ones((1,)), obs, np.ones((1,),
                                                            dtype=np.bool_))


class ReplayBuffer(object):
  def __init__(self,
               init_transition: TransitionTuple,
               capacity: int = 1000000) -> None:
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
