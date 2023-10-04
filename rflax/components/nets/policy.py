"""Policy networks."""

from ..blocks import MlpBlock, MlpConfig
from ...types import Array

import flax.linen as nn
from typing import Tuple


class DetTanhPolicy(nn.Module):
  action_dim: int
  config: MlpConfig

  @nn.compact
  def __call__(self, observation: Array) -> Array:
    return nn.tanh(MlpBlock(self.action_dim, self.config, name="det_pi")(observation))


class NormalPolicy(nn.Module):
  action_dim: int
  config: MlpConfig

  @nn.compact
  def __call__(self, observation: Array) -> Tuple[Array, Array]:
    outp = MlpBlock(self.action_dim * 2, self.config, name="normal_pi")(observation)
    outp = outp.reshape(*outp.shape[:-1], self.action_dim, 2)
    mean, logstd = outp[..., 0], outp[..., 1]

    return mean, logstd
