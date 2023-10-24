"""Basic NN networks."""

from ..types import Array, DType, Initializer
from .initializers import kernel_default, bias_default

from jutils import np

import flax.linen as nn
from flax import struct


class ScalarBlock(nn.Module):
  init_value: float

  def setup(self):
    self.value = self.param("value", lambda x: self.init_value)

  def __call__(self):
    return self.value


@struct.dataclass
class MlpConfig:
  hidden_dims: str = "256-256"
  act_fn: str = "tanh"
  dtype: DType = np.float_
  use_bias: bool = True
  kernel_init: Initializer = kernel_default()
  bias_init: Initializer = bias_default()


def build_dense(n_feats: int, config: MlpConfig, name: str) -> nn.Module:
  return nn.Dense(
      features=n_feats,
      use_bias=config.use_bias,
      dtype=config.dtype,
      param_dtype=config.dtype,
      kernel_init=config.kernel_init,
      bias_init=config.bias_init,
      name=name,
  )


class MlpBlock(nn.Module):
  out_dim: int
  config: MlpConfig

  @nn.compact
  def __call__(self, inp: Array) -> Array:
    config = self.config

    for i, feats in enumerate(config.hidden_dims.split("-")):
      lyr_name = f"hidden_{i}"
      inp = build_dense(int(feats), config, lyr_name)(inp)
      inp = getattr(nn, config.act_fn)(inp)

    return build_dense(self.out_dim, config, "out_layer")(inp)
