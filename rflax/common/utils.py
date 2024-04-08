from typing import TypeVar

import chex as cx
import jax
import jax.random as random

PyTree = TypeVar("PyTree")


def construct_minibatches_and_shuffle(key: cx.PRNGKey, batch: PyTree, n_mb: int,
                                      mb_size: int) -> PyTree:
  size = n_mb * mb_size
  inds = random.permutation(key, size)
  return jax.tree_map(lambda x: x[inds].reshape(n_mb, -1, *x.shape[1:]), batch)
