"""Policy networks."""

from rflax.components.blocks import MlpBlock, MultiOutputMlp
from rflax.components.initializers import kernel_default, bias_default
from rflax.types import Array, DType, Initializer

import chex
import distrax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class DetTanhPolicy(nn.Module):
  action_dim: int
  hidden_dim: int = 2048
  dtype: DType = jnp.float32
  activations: str = "relu"
  kernel_init: Initializer = kernel_default()
  bias_init: Initializer = bias_default()
  intermediate_dropout: float = 0.1
  final_dropout: Optional[float] = None

  @nn.compact
  def __call__(self,
               observations: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    action = MlpBlock(
        out_dim=self.action_dim,
        use_bias=True,
        intermediate_dim=self.hidden_dim,
        dtype=self.dtype,
        activations=self.activations,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        intermediate_dropout=self.intermediate_dropout,
        final_dropout=self.final_dropout,
        name="det_tanh_policy",
    )(observations, enable_dropout)
    return nn.tanh(action)


class NormalPolicy(nn.Module):
  action_dim: int
  hidden_dim: int = 2048
  dtype: DType = jnp.float32
  activations: str = "relu"
  kernel_init: Initializer = kernel_default()
  bias_init: Initializer = bias_default()
  intermediate_dropout: float = 0.1
  final_dropout: Optional[float] = None

  @nn.compact
  def __call__(self,
               observations: Array,
               enable_dropout: bool = True) -> distrax.Distribution:
    mean, logstd = MultiOutputMlp(
        out_dim=(self.action_dim,) * 2,
        use_bias=True,
        intermediate_dim=self.hidden_dim,
        dtype=self.dtype,
        activations=self.activations,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        intermediate_dropout=self.intermediate_dropout,
        final_dropout=self.final_dropout,
        name="normal_policy",
    )(observations, enable_dropout)
    return distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(logstd))
