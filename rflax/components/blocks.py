"""Basic nn blocks."""

from rflax.types import Array, DType, Initializer

from typing import Optional, Union, Callable, Sequence

import jax.numpy as jnp
import flax.linen as nn


def _convert_to_activation_fn(fn_or_string: Union[str, Callable]) -> Callable:
  if fn_or_string == "linear":
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    return ValueError(
        "Don't know how to convert %s to an activation function." %
        (fn_or_string,))


class MlpBlock(nn.Module):
  out_dim: int
  use_bias: bool
  intermediate_dim: int = 2048
  dtype: DType = jnp.float32
  activations: Sequence[Union[str, Callable]] = ("relu",)
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  intermediate_dropout: float = 0.1
  final_dropout: Optional[float] = None

  @nn.compact
  def __call__(self, inp: Array, enable_dropout: bool = True) -> Array:
    def dense(n_feats: int, name: str, inputs: Array, dropout: float) -> Array:
      x = nn.Dense(
          features=n_feats,
          use_bias=self.use_bias,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          name=name,
      )(inputs)
      return nn.Dropout(rate=dropout)(x, deterministic=not enable_dropout)

    for i, act_fn in enumerate(self.activations):
      dense_name = "hidden" if len(self.activations) == 1 else f"hidden_{i}"
      inp = dense(self.intermediate_dim, dense_name, inp,
                  self.intermediate_dropout)
      inp = _convert_to_activation_fn(act_fn)(inp)

    return dense(
        self.out_dim,
        "out",
        inp,
        self.final_dropout if self.final_dropout else self.intermediate_dropout,
    )
