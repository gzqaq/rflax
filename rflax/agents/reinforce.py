"""Implementation of REINFORCE."""

from rflax.agents import ContinuousAgent
from rflax.components.nets.policy import NormalPolicy
from rflax.components.initializers import kernel_default, bias_default
from rflax.utils import init_model
from rflax.types import Array, MetricDict, PRNGKey, ConfigDictLike

import chex
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from functools import partial
from ml_collections import ConfigDict
from typing import Tuple, Optional, Sequence


@partial(jax.jit, static_argnames=("enable_dropout",))
def _actor_apply(
    rng: PRNGKey, actor: TrainState, observations: Array,
    enable_dropout: bool) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
  dropout_rng, sample_rng = jax.random.split(rng)
  return actor.apply_fn(
      {
          "params": actor.params
      },
      observations,
      enable_dropout,
      rngs={
          "dropout": dropout_rng
      },
  ).sample_and_log_prob(seed=sample_rng)


@partial(jax.jit, static_argnames=("enable_dropout",))
def _log_prob_of_action(
    rng: PRNGKey,
    actor: TrainState,
    observations: Array,
    actions: Array,
    enable_dropout: bool,
) -> chex.ArrayDevice:
  return actor.apply_fn({
      "params": actor.params
  },
                        observations,
                        enable_dropout,
                        rngs={
                            "dropout": rng
                        }).log_prob(actions)


@jax.jit
def _update_actor(rng: PRNGKey, actor: TrainState, batch: FrozenDict,
                  discount: float) -> Tuple[TrainState, MetricDict]:
  calc_return = lambda carry, rew: (carry * discount + rew, carry * discount +
                                    rew)
  ret = jnp.flip(
      jax.lax.scan(calc_return, jnp.zeros((1,)), jnp.flip(batch["rewards"]))[1])

  # calc_grad = lambda ret, grad: jax.tree_map(lambda x: -x * ret, grad)
  # is_leaf = lambda x: isinstance(x, FrozenDict)
  # grads = jax.tree_map(calc_grad, list(ret), batch["grads"], is_leaf=is_leaf)
  # grads = jax.tree_util.tree_reduce(lambda x, y: jax.tree_map(jnp.add, x, y), grads, is_leaf=is_leaf)
  # loss = jnp.sum(-ret * batch["log_prob"])

  def loss_fn(params):
    logprobs = _log_prob_of_action(
        rng,
        actor.replace(params=params),
        batch["observations"],
        batch["actions"],
        True,
    )
    loss = -logprobs * ret

    return jnp.sum(loss)

  loss, grads = jax.value_and_grad(loss_fn)(actor.params)
  return actor.apply_gradients(grads=grads), {"loss": loss}


class REINFORCE(ContinuousAgent):
  @staticmethod
  def get_default_config(
      updates: Optional[ConfigDictLike] = None) -> ConfigDict:
    config = ConfigDict()
    config.discount = 0.98
    config.lr = 0.0001

    config.mlp_args = ConfigDict()
    config.mlp_args.hidden_dim = 256
    config.mlp_args.activations = "relu-relu"
    config.mlp_args.dtype = jnp.float32
    config.mlp_args.kernel_init = kernel_default()
    config.mlp_args.bias_init = bias_default()
    config.mlp_args.intermediate_dropout = 0  # TODO: dropout makes actor fail
    # to retrieve accurate log prob of action when calculating grads; the comm-
    # ented code (line 34-38) doesn't suffer this problem but is slow
    config.mlp_args.final_dropout = None

    if updates:
      config.update(ConfigDict(updates).copy_and_resolve_references())

    return config

  def __init__(
      self,
      config: ConfigDict,
      rng: PRNGKey,
      obs_dim: int,
      action_bound: Sequence[Array],
  ) -> None:
    super().__init__(config, obs_dim, action_bound)

    rng, init_rng = jax.random.split(rng)
    fake_obs = jnp.zeros((1, self.obs_dim))
    actor = NormalPolicy(self.action_dim, **self.config.mlp_args.to_dict())
    actor_params = init_model(actor, init_rng, fake_obs)["params"]
    tx = optax.chain(optax.clip_by_global_norm(1.01),
                     optax.adam(self.config.lr))
    self._actor = TrainState.create(apply_fn=actor.apply,
                                    params=actor_params,
                                    tx=tx)

    self._rng = rng

  def sample_actions(self, observations: Array) -> chex.ArrayDevice:
    self._rng, dropout_rng = jax.random.split(self._rng)
    a, _ = _actor_apply(dropout_rng, self._actor, observations, True)

    return a

  def eval_actions(self, observations: Array) -> chex.ArrayDevice:
    action, _ = _actor_apply(self._rng, self._actor, observations, False)

    return action

  def update(self, batch: FrozenDict) -> MetricDict:
    self._rng, update_rng = jax.random.split(self._rng)
    self._actor, metrics = _update_actor(update_rng, self._actor, batch,
                                         self.config.discount)
    self._step += 1

    chex.assert_tree_all_finite(
        metrics)  # TODO: loss becomes too small when training

    return metrics

  def save_checkpoint(
      self,
      ckpt_dir: str,
      prefix: str = "reinforce_ckpt_",
      keep: int = 1,
      overwrite: bool = False,
      keep_every_n_steps: Optional[int] = None,
  ) -> None:
    config_dict = {
        "_obs_dim": self.obs_dim,
        "_action_dim": self.action_dim,
        "action_high": jax.device_get(self.action_high),
        "action_low": jax.device_get(self.action_low),
        "_rng": jax.device_get(self._rng),
        "_step": self.step,
    }
    params_dict = {
        "actor": self._actor,
    }

    save_checkpoint(
        ckpt_dir,
        FrozenDict({
            "config": config_dict,
            "params": params_dict
        }),
        self.step,
        prefix,
        keep,
        overwrite,
        keep_every_n_steps,
    )

  def restore_checkpoint(self,
                         ckpt_dir,
                         prefix: str = "reinforce_ckpt_",
                         step: Optional[int] = None) -> None:
    state_dict = FrozenDict(restore_checkpoint(ckpt_dir, None, step, prefix))

    for k, v in state_dict["config"].items():
      setattr(self, k, jax.device_put(v) if isinstance(v, Array) else v)

    self._actor = self._actor.replace(**state_dict["params"]["actor"])
