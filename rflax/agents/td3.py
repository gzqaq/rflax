from ..types import VariableDict

from flax import struct


@struct.dataclass
class TD3State:
  actor_params: VariableDict
  critic_params: VariableDict
  target_actor_params: VariableDict
  target_critic_params: VariableDict
