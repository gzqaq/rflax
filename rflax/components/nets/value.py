"""Value networks."""

from rflax.components.blocks import MlpBlock
from rflax.components.initializers import kernel_default, bias_default
from rflax.types import Array, DType, Initializer
from rflax.utils import expand_and_repeat

import chex
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class StateValue(nn.Module):
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
    critic = MlpBlock(
        out_dim=1,
        use_bias=True,
        intermediate_dim=self.hidden_dim,
        dtype=self.dtype,
        activations=self.activations,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        intermediate_dropout=self.intermediate_dropout,
        final_dropout=self.final_dropout,
        name="state_value",
    )(observations, enable_dropout)

    return critic


class StateActionValue(nn.Module):
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
               actions: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    chex.assert_equal_rank([observations, actions])
    chex.assert_equal_shape_prefix([observations, actions], -1)

    inp = jnp.concatenate([observations, actions], axis=-1)
    critic = MlpBlock(
        out_dim=1,
        use_bias=True,
        intermediate_dim=self.hidden_dim,
        dtype=self.dtype,
        activations=self.activations,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        intermediate_dropout=self.intermediate_dropout,
        final_dropout=self.final_dropout,
        name="state_action_value",
    )(inp, enable_dropout)

    return critic


class StateMultiActionValue(nn.Module):
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
               actions: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    chex.assert_rank(observations, actions.ndim - 1)

    observations = expand_and_repeat(observations, 1, actions.shape[1])

    inp = jnp.concatenate([observations, actions], axis=-1)
    critic = MlpBlock(
        out_dim=1,
        use_bias=True,
        intermediate_dim=self.hidden_dim,
        dtype=self.dtype,
        activations=self.activations,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        intermediate_dropout=self.intermediate_dropout,
        final_dropout=self.final_dropout,
        name="state_action_value",
    )(inp, enable_dropout)

    return critic
