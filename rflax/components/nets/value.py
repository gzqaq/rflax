"""Value networks."""

from rflax.components.blocks import MlpBlock, MlpConfig
from rflax.components.initializers import kernel_default, bias_default
from rflax.types import Array, DType, Initializer
from rflax.utils import expand_and_repeat

import chex
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class StateValue(nn.Module):
  config: MlpConfig

  @nn.compact
  def __call__(self,
               observations: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    critic = MlpBlock(
        out_dim=1,
        use_bias=True,
        config=self.config,
        name="state_value",
    )(observations, enable_dropout)

    return critic


class StateDiscreteActionValue(nn.Module):
  n_actions: int
  config: MlpConfig

  @nn.compact
  def __call__(self,
               observations: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    critic = MlpBlock(
        out_dim=self.n_actions,
        use_bias=True,
        config=self.config,
        name="state_action_value_by_action",
    )(observations, enable_dropout)

    return critic


class StateActionValue(nn.Module):
  config: MlpConfig

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
        config=self.config,
        name="state_action_value",
    )(inp, enable_dropout)

    return critic


class StateMultiActionValue(nn.Module):
  config: MlpConfig

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
        config=self.config,
        name="state_action_value",
    )(inp, enable_dropout)

    return critic


class StateActionEnsemble(nn.Module):
  num_qs: int
  config: MlpConfig

  @nn.compact
  def __call__(self,
               observations: Array,
               actions: Array,
               enable_dropout: bool = True) -> chex.ArrayDevice:
    vmap_critic = nn.vmap(
        StateActionValue,
        variable_axes={"params": 0},
        split_rngs={
            "params": True,
            "dropout": True
        },
        in_axes=None,
        out_axes=0,
        axis_size=self.num_qs,
    )
    qs = vmap_critic(config=self.config)(observations, actions, enable_dropout)

    return qs
