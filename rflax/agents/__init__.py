"""Implementations of RL agents."""

# from rflax.types import Array, ConfigDictLike, VariableDict

# import chex
# import jax
# from copy import deepcopy
# from ml_collections import ConfigDict
# from typing import Optional, Sequence

# @chex.dataclass(frozen=True)
# class AgentConfig:
#   pass

# class Agent(object):
#   @staticmethod
#   def default_config():
#     raise NotImplementedError

#   def __init__(self, config, obs_dim: int) -> None:
#     self.config = config
#     self._obs_dim = obs_dim
#     self._step = 0

#   @property
#   def obs_dim(self) -> int:
#     return self._obs_dim

#   @property
#   def step(self) -> int:
#     return self._step

# class DiscreteAgent(Agent):
#   def __init__(self, config, obs_dim: int, n_actions: int) -> None:
#     super().__init__(config, obs_dim)

#     self._n_actions = n_actions

#   @property
#   def n_actions(self) -> int:
#     return self._n_actions

# class ContinuousAgent(Agent):
#   def __init__(self, config, obs_dim: int,
#                action_bound: Sequence[Array]) -> None:
#     super().__init__(config, obs_dim)

#     self.action_high = jax.device_put(action_bound[0])
#     self.action_low = (-action_bound[0] if len(action_bound) == 1 else
#                        jax.device_put(action_bound[1]))
#     self._action_dim = self.action_high.shape[0]

#   @property
#   def action_dim(self) -> int:
#     return self._action_dim

# class TargetParams(object):
#   """Store params of target networks."""
#   def __init__(self, **kwargs):
#     self._keys = []
#     for k, v in kwargs.items():
#       setattr(self, k, deepcopy(v))
#       self._keys.append(k)

#   def to_dict(self) -> VariableDict:
#     return {k: getattr(self, k) for k in self._keys}
