"""Value networks."""

from ..blocks import MlpBlock, MlpConfig
from ...types import Array

from jutils import np

import flax.linen as nn


class StateValue(nn.Module):
  config: MlpConfig

  @nn.compact
  def __call__(self, state: Array) -> Array:
    return MlpBlock(1, self.config, name="state_val")(state)


class DiscreteActionValue(nn.Module):
  n_actions: int
  config: MlpConfig

  @nn.compact
  def __call__(self, state: Array) -> Array:
    return MlpBlock(self.n_actions, self.config, name="action_val")(state)


class ContinuousActionValue(nn.Module):
  config: MlpConfig

  @nn.compact
  def __call__(self, state: Array, action: Array) -> Array:
    inp = np.concatenate([state, action], axis=-1)
    return MlpBlock(1, self.config, name="action_val")(inp)


class ActionValueEnsemble(nn.Module):
  n_qs: int
  config: MlpConfig

  @nn.compact
  def __call__(self, state: Array, action: Array) -> Array:
    return nn.vmap(
        ContinuousActionValue,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=self.n_qs,
    )(self.config)(state, action)
