"""Policy networks."""

from ..blocks import MlpBlock, MlpConfig, ScalarBlock
from ...types import Array

from jutils import np

import distrax
import flax.linen as nn
from typing import Tuple


class DetTanhPolicy(nn.Module):
  action_dim: int
  config: MlpConfig

  @nn.compact
  def __call__(self, observation: Array) -> Array:
    return nn.tanh(
        MlpBlock(self.action_dim, self.config, name="det_pi")(observation))


class NormalPolicy(nn.Module):
  action_dim: int
  config: MlpConfig

  @nn.compact
  def __call__(self, observation: Array) -> Tuple[Array, Array]:
    outp = MlpBlock(self.action_dim * 2, self.config,
                    name="normal_pi")(observation)
    outp = outp.reshape(*outp.shape[:-1], self.action_dim, 2)
    mean, logstd = outp[..., 0], outp[..., 1]

    return mean, logstd


def tanh_gaussian_dist(mean: Array, log_std: Array) -> distrax.Distribution:
  return distrax.Transformed(
      distrax.MultivariateNormalDiag(mean, np.exp(log_std)),
      distrax.Block(distrax.Tanh(), ndims=1),
  )


class TanhGaussianPolicy(nn.Module):
  action_dim: int
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  config: MlpConfig = MlpConfig()

  def setup(self):
    self.base = MlpBlock(self.action_dim * 2, self.config, name="pi_base")
    self.multiplier = ScalarBlock(self.log_std_multiplier)
    self.offset = ScalarBlock(self.log_std_offset)

  def log_prob(self, observation: Array, action: Array) -> Array:
    outp = self.base(observation)
    mean, log_std = np.split(outp, 2, axis=-1)
    log_std = self.multiplier() * log_std + self.offset()
    log_std = np.clip(log_std, -20, 2)

    return tanh_gaussian_dist(mean, log_std).log_prob(action)

  def __call__(self, observation: Array) -> Array:
    outp = self.base(observation)
    mean, log_std = np.split(outp, 2, axis=-1)
    log_std = self.multiplier() * log_std + self.offset()
    log_std = np.clip(log_std, -20, 2)

    return mean, log_std
