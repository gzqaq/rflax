"""Useful utility functions."""

from rflax.types import PRNGKey, VariableDict, DataDict

import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.core.frozen_dict import FrozenDict
from typing import Callable, Any, Union, List


def expand_and_repeat(array: chex.Array, axis: int,
                      repeats: int) -> chex.ArrayDevice:
  array = jnp.expand_dims(array, axis=axis)
  return jnp.repeat(array, repeats=repeats, axis=axis)


def to_jax_batch(array: chex.Array) -> chex.ArrayBatched:
  """Add one dimension outside the input and convert it to jax.Array."""
  return jax.device_put(array[None, ...])


def batch_to_jax(batch: DataDict) -> FrozenDict:
  return FrozenDict(jax.tree_map(jax.device_put, batch))


def squeeze_to_np(array: chex.ArrayDevice) -> np.ndarray:
  """Convert a jax.Array to np.ndarray and squeeze its first dimension."""
  return np.array(array).squeeze(axis=0)


def init_model(model: nn.Module, rng: PRNGKey, *args) -> VariableDict:
  param_rng, dropout_rng = jax.random.split(rng)

  return model.init({"params": param_rng, "dropout": dropout_rng}, *args)


def model_apply(rng: PRNGKey, params: VariableDict, apply_fn: Callable,
                *args) -> Any:
  return apply_fn({"params": params}, *args, rngs={"dropout": rng})


def soft_target_update(src: VariableDict, target: VariableDict,
                       tau: float) -> VariableDict:
  return jax.tree_map(lambda s, t: s * tau + t * (1 - tau), src, target)


def _insert_record(storage: DataDict, record: DataDict,
                   ind: Union[np.ndarray, int]) -> DataDict:
  for k in storage.keys():
    storage[k][ind] = record[k]

  return storage


class ReplayBuffer(object):
  def __init__(self, capacity: int, obs_dim: int, action_dim: int) -> None:
    self._storage = {
        "observations": np.empty((capacity, obs_dim), dtype=np.float32),
        "actions": np.empty((capacity, action_dim), dtype=np.float32),
        "rewards": np.empty((capacity, 1), dtype=np.float32),
        "next_observations": np.empty((capacity, obs_dim), dtype=np.float32),
        "masks": np.empty((capacity, 1), dtype=np.float32),
        "dones": np.empty((capacity, 1), dtype=bool),
    }

    self._capacity = capacity
    self._size = 0
    self._insert_ind = 0

  def insert(self, record: DataDict) -> None:
    self._storage = _insert_record(self._storage, record, self._insert_ind)
    self._insert_ind = (self._insert_ind + 1) % self.capacity
    self._size = min(self.size + 1, self.capacity)

  def insert_batch(self, batch: DataDict) -> None:
    bs = len(batch["observations"])
    inds = np.arange(self._insert_ind, self._insert_ind + bs) % self.capacity

    self._storage = _insert_record(self._storage, batch, inds)
    self._insert_ind = (self._insert_ind + bs) % self.capacity
    self._size = min(self.size + bs, self._capacity)

  def sample(self, batch_size: int) -> DataDict:
    inds = np.arange(self._size)
    np.random.shuffle(inds)
    inds = inds[:batch_size]

    return {k: v[inds] for k, v in self._storage.items()}

  @property
  def capacity(self) -> int:
    return self._capacity

  @property
  def size(self) -> int:
    return self._size

  @property
  def keys(self) -> List[str]:
    return list(self._storage.keys())

  def __len__(self) -> int:
    return self.size
