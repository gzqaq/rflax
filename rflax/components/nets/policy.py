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


class TanhMultivariateNormalDiag(distrax.Transformed):
  def __init__(
      self,
      loc: Array,
      scale_diag: Array,
      low: Optional[Array] = None,
      high: Optional[Array] = None,
  ) -> None:
    distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                  scale_diag=scale_diag)

    layers = []

    if not (low is None or high is None):

      def rescale_from_tanh(x):
        x = (x + 1) / 2  # (-1, 1) => (0, 1)
        return x * (high - low) + low

      def forward_log_det_jacobian(x):
        high_ = jnp.broadcast_to(high, x.shape)
        low_ = jnp.broadcast_to(low, x.shape)

        return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

      layers.append(
          distrax.Lambda(
              rescale_from_tanh,
              forward_log_det_jacobian=forward_log_det_jacobian,
              event_ndims_in=1,
              event_ndims_out=1,
          ))

    layers.append(distrax.Block(distrax.Tanh(), 1))
    bijector = distrax.Chain(layers)

    super().__init__(distribution=distribution, bijector=bijector)

  def mode(self) -> chex.ArrayDevice:
    return self.bijector.forward(self.distribution.mode())


class NormalTanhPolicy(nn.Module):
  action_dim: int
  high: Optional[Array] = None
  low: Optional[Array] = None
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
        name="normal_tanh_policy",
    )(observations, enable_dropout)
    return TanhMultivariateNormalDiag(loc=mean,
                                      scale_diag=jnp.exp(logstd),
                                      low=self.low,
                                      high=self.high)
