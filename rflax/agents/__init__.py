"""Implementations of RL agents."""

from rflax.types import Array, PRNGKey, ConfigDictLike

from copy import deepcopy
from ml_collections import ConfigDict
from typing import Optional, Sequence


class Agent(object):
  @staticmethod
  def get_default_config(
      updates: Optional[ConfigDictLike] = None) -> ConfigDict:
    raise NotImplementedError

  def __init__(self, config: ConfigDict, obs_dim: int,
               action_bound: Sequence[Array]) -> None:
    self.config = self.get_default_config(config)

    self.action_high = action_bound[0]
    self.action_low = (-action_bound[0]
                       if len(action_bound) == 1 else action_bound[1])

    self._obs_dim = obs_dim
    self._action_dim = self.action_high.shape[0]

  @property
  def obs_dim(self) -> int:
    return self._obs_dim

  @property
  def action_dim(self) -> int:
    return self._action_dim


class TargetParams(object):
  """Store params of target networks."""
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, deepcopy(v))
