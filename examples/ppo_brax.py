import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import brax.envs as envs
import chex as cx
import flax.serialization as serl
import jax
import jax.lax as lax
import jax.numpy as np
import jax.random as random
from flax.training.train_state import TrainState
from brax.envs.base import Env as BraxEnv, State as EnvState

from rflax.common.dist import TanhNormal
from rflax.ppo.train import init_train
from rflax.ppo.update import RolloutSamples, update_on_samples
from rflax.type_hints import Variable


class Args(NamedTuple):
  seed: int = 42
  n_seeds: int = 20
  log_dir: str = "logs/ppo"
  env_name: str = "humanoid"
  total_timesteps: int = 1_000_000
  ep_len: int = 1_000
  rollout_steps: int = 2_048
  n_epochs: int = 10
  n_minibatches: int = 32
  lr: float = 3e-4
  anneal_lr: bool = True
  clip_norm: float = 0.5
  log_std_init: float = 0.0
  discount: float = 0.99
  gae_lambda: float = 0.95
  clip_eps: float = 0.2
  clip_vf: float | None = None
  ent_coef: float = 0.0
  vf_coef: float = 0.5
  hidden_dim: int = 64
  act_fn: str = "tanh"


class Metrics(NamedTuple):
  eval_return: cx.Array
  policy_loss: cx.Array
  value_loss: cx.Array
  entropy: cx.Array
  old_approx_kl: cx.Array
  approx_kl: cx.Array
  clip_frac: cx.Array


def parse_args() -> Args:
  parser = argparse.ArgumentParser()
  default_args = Args()

  for k, v in default_args._asdict().items():
    opt = f"--{k.replace('_', '-')}"
    if isinstance(v, int):
      parser.add_argument(opt, type=int, default=v)
    elif isinstance(v, float) or v is None:
      parser.add_argument(opt, type=float, default=v)
    elif isinstance(v, bool):
      parser.add_argument(opt, type=bool, default=v)
    else:
      parser.add_argument(opt, type=str, default=v)

  return Args(**vars(parser.parse_args()))


def make_train(
    args: Args,
    env: BraxEnv,
) -> Callable[[cx.PRNGKey], tuple[TrainState, Metrics]]:
  env = envs.training.AutoResetWrapper(
      env=envs.training.EpisodeWrapper(
          env=env,
          episode_length=args.ep_len,
          action_repeat=1,
      ),)
  eval_env = envs.training.VmapWrapper(
      env=envs.training.EvalWrapper(env),
      batch_size=3,
  )

  n_updates = args.total_timesteps // args.rollout_steps
  init_fn = init_train(
      n_epochs=args.n_epochs,
      n_minibatches=args.n_minibatches,
      n_updates=n_updates,
      action_dim=env.action_size,
      log_std_init=args.log_std_init,
      hidden_dim=args.hidden_dim,
      act_fn=args.act_fn,
      lr=args.lr,
      anneal_lr=args.anneal_lr,
      clip_norm=args.clip_norm,
  )
  update_fn = update_on_samples(
      rollout_steps=args.rollout_steps,
      n_epochs=args.n_epochs,
      n_minibatches=args.n_minibatches,
      clip_eps=args.clip_eps,
      clip_vf=args.clip_vf,
      vf_coef=args.vf_coef,
      ent_coef=args.ent_coef,
      discount=args.discount,
      gae_lambda=args.gae_lambda,
  )

  def train(key: cx.PRNGKey) -> tuple[TrainState, Metrics]:
    keys = random.split(key, 3)
    env_state = env.reset(keys[0])
    agent = init_fn(keys[1], np.ones((env.observation_size,)))
    apply_fn = agent.apply_fn

    # iterations
    def it_run(
        carry: tuple[cx.PRNGKey, TrainState, EnvState],
        x: None,
    ) -> tuple[tuple[cx.PRNGKey, TrainState, EnvState], Metrics]:

      def rollout(
          carry: tuple[cx.PRNGKey, Variable, EnvState],
          x: None,
      ) -> tuple[tuple[cx.PRNGKey, Variable, EnvState], RolloutSamples]:
        key, params, env_state = carry
        s = env_state.obs
        pi: TanhNormal
        pi, val = apply_fn(params, s)
        key, sk = random.split(key)
        a, log_prob = pi.sample_and_log_prob(seed=sk)
        env_state = env.step(env_state, a)

        return (key, params, env_state), RolloutSamples(
            s=s,
            a=a,
            log_prob=log_prob,
            val=val,
            s_=env_state.obs,
            r=env_state.reward,
            d=env_state.done,
        )

      key, agent, env_state = carry
      (key, _, env_state), samples = lax.scan(
          f=rollout,
          init=(key, agent.params, env_state),
          xs=None,
          length=args.rollout_steps,
      )
      keys = random.split(key, 3)
      agent, losses = update_fn(keys[0], agent, samples)

      # eval
      def eval_rollout(
          carry: tuple[cx.PRNGKey, Variable, EnvState],
          x: None,
      ) -> tuple[tuple[cx.PRNGKey, Variable, EnvState], None]:
        key, params, eval_state = carry
        pi: TanhNormal
        pi, val = apply_fn(params, eval_state.obs)
        key, sk = random.split(key)
        a = pi.sample(seed=sk)
        env_state = eval_env.step(eval_state, a)

        return (key, params, env_state), None

      eval_state = eval_env.reset(keys[1])
      (key, _, eval_state), _ = lax.scan(
          f=eval_rollout,
          init=(keys[2], agent.params, eval_state),
          xs=None,
          length=args.ep_len,
      )
      metrics = Metrics(
          eval_return=eval_state.info["eval_metrics"].episode_metrics["reward"]
          .mean(),
          **losses._asdict(),
      )

      return (key, agent, env_state), metrics

    (_, agent, _), metrics = lax.scan(
        f=it_run,
        init=(keys[2], agent, env_state),
        xs=None,
        length=n_updates,
    )

    return agent, metrics

  return train


def core(args: Args) -> None:
  env = envs.get_environment(args.env_name)
  train_fn = jax.vmap(make_train(args, env))
  keys = random.split(random.key(args.seed), args.n_seeds)

  agents, metrics = train_fn(keys)

  log_dir = Path(args.log_dir).expanduser().resolve()
  if not log_dir.exists():
    log_dir.mkdir(parents=True)

  with open(log_dir / "metrics.bin", "wb") as fd:
    fd.write(serl.to_bytes(metrics))
  with open(log_dir / "ckpt.bin", "wb") as fd:
    fd.write(serl.to_bytes(agents))


if __name__ == "__main__":
  core(parse_args())
