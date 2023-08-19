"""Policy networks."""

from rflax.components.blocks import MlpBlock, MultiOutputMlp, MlpConfig
from rflax.components.initializers import kernel_default, bias_default
from rflax.types import Array, DType, Initializer

import chex
import distrax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class DetTanhPolicy(nn.Module):
  action_dim: int
  config: MlpConfig

  @nn.compact
  def __call__(self,
               observations: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    action = MlpBlock(
        out_dim=self.action_dim,
        use_bias=True,
        config=self.config,
        name="det_tanh_policy",
    )(observations, enable_dropout)
    return nn.tanh(action)


class NormalPolicy(nn.Module):
  action_dim: int
  config: MlpConfig

  @nn.compact
  def __call__(self,
               observations: Array,
               enable_dropout: bool = True) -> distrax.Distribution:
    mean, logstd = MultiOutputMlp(
        out_dim=(self.action_dim,) * 2,
        use_bias=True,
        config=self.config,
        name="normal_policy",
    )(observations, enable_dropout)
    return distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(logstd))


class NormalTanhPolicy(nn.Module):
  action_dim: int
  high: Array
  low: Array
  config: MlpConfig

  @nn.compact
  def __call__(self,
               observations: Array,
               enable_dropout: bool = True) -> distrax.Distribution:
    mean, logstd = MultiOutputMlp(
        out_dim=(self.action_dim,) * 2,
        use_bias=True,
        config=self.config,
        name="normal_tanh_policy",
    )(observations, enable_dropout)
    mean = jnp.tanh(mean)
    mean = jnp.where(mean > 0, mean * self.high, -mean * self.low)
    return distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(logstd))
