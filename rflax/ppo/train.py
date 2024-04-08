from collections.abc import Callable

import chex as cx
import jax
import optax as ox
from flax.training.train_state import TrainState

from rflax.ppo.net import ActorCritic


def init_train(
    n_epochs: int,
    n_minibatches: int,
    n_updates: int,
    action_dim: int,
    log_std_init: float,
    hidden_dim: int,
    act_fn: str,
    lr: float,
    anneal_lr: bool,
    clip_norm: float,
) -> Callable[[cx.PRNGKey, jax.Array], TrainState]:

  def fn(key: cx.PRNGKey, init_s: jax.Array) -> TrainState:
    # init agent
    agent = ActorCritic(action_dim, log_std_init, hidden_dim, act_fn)
    params = agent.init(key, init_s)

    # init train state
    tx = ox.chain(
        ox.clip_by_global_norm(clip_norm),
        ox.adam(
            learning_rate=lambda step:
            (lr * (1 - step // (n_minibatches * n_epochs) / n_updates)
             if anneal_lr else lr),
            eps=1e-7,
        ),
    )

    return TrainState.create(apply_fn=agent.apply, params=params, tx=tx)

  return fn
