"""Basic nn blocks."""

from rflax.types import Array, DType, Initializer, ActivationFn
from rflax.components.initializers import kernel_default, bias_default

from typing import Optional, Union, Sequence

import chex
import jax.numpy as jnp
import flax.linen as nn


def _convert_to_activation_fn(
    fn_or_string: Union[str, ActivationFn]) -> ActivationFn:
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("Don't know how to convert %s to an activation function." %
                     (fn_or_string,))

@chex.dataclass(frozen=True)
class MlpConfig:
  dtype: DType = jnp.float32
  activations: str = "relu"
  intermediate_dim: int = 2048
  kernel_init: Initializer = kernel_default()
  bias_init: Initializer = bias_default()
  intermediate_dropout: float = 0.1
  final_dropout: Optional[float] = None


class MlpBlock(nn.Module):
  out_dim: int
  use_bias: bool
  config: MlpConfig

  @nn.compact
  def __call__(self,
               inp: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    config = self.config

    def dense(n_feats: int, name: str, inputs: Array,
              dropout: float) -> chex.ArrayDevice:
      x = nn.Dense(
          features=n_feats,
          use_bias=self.use_bias,
          dtype=config.dtype,
          kernel_init=config.kernel_init,
          bias_init=config.bias_init,
          name=name,
      )(inputs)
      return nn.Dropout(rate=dropout)(x, deterministic=not enable_dropout)

    for i, act_fn in enumerate(config.activations.split("-")):
      dense_name = "hidden" if len(config.activations) == 1 else f"hidden_{i}"
      inp = dense(config.intermediate_dim, dense_name, inp,
                  config.intermediate_dropout)
      inp = _convert_to_activation_fn(act_fn)(inp)

    return dense(
        self.out_dim,
        "out",
        inp,
        config.final_dropout if config.final_dropout else config.intermediate_dropout,
    )


class MultiOutputMlp(nn.Module):
  out_dim: Sequence[int]
  use_bias: bool
  config: MlpConfig

  @nn.compact
  def __call__(self,
               inp: Array,
               enable_dropout: bool = True) -> Sequence[Array]:
    config = self.config

    def dense(n_feats: int, name: str, inputs: Array,
              dropout: float) -> chex.ArrayDevice:
      x = nn.Dense(
          features=n_feats,
          use_bias=self.use_bias,
          dtype=config.dtype,
          kernel_init=config.kernel_init,
          bias_init=config.bias_init,
          name=name,
      )(inputs)
      return nn.Dropout(rate=dropout)(x, deterministic=not enable_dropout)

    for i, act_fn in enumerate(config.activations.split("-")):
      dense_name = "hidden" if len(config.activations) == 1 else f"hidden_{i}"
      inp = dense(config.intermediate_dim, dense_name, inp,
                  config.intermediate_dropout)
      inp = _convert_to_activation_fn(act_fn)(inp)

    outputs = []
    do_rate = (config.final_dropout
               if config.final_dropout else config.intermediate_dropout)
    for i, od in enumerate(self.out_dim):
      outputs.append(
          dense(od, "out" if len(self.out_dim) == 1 else f"out_{i}", inp,
                do_rate))

    return tuple(outputs)
