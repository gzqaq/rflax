from rflax.agents import sac, cql
from rflax.utils import TransitionTuple
from rflax.logging import WandBLogger

from jutils import jax, lax, np, random, rng_wrapper

import gymnasium as gym
import hydra
import optax
from hydra.utils import instantiate
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg.sac))
  logger = WandBLogger(instantiate(cfg.sac.logging),
                       OmegaConf.to_container(cfg.sac))

  env = gym.make(cfg.cql.env)
  conf = instantiate(cfg.sac.train)
  cql_conf = instantiate(cfg.cql.train)
  mlp_conf = instantiate(cfg.sac.mlp)

  ds = np.load(Path(cfg.cql.npz_path).expanduser())
  obs = ds["s"]
  a = ds["a"]
  r = ds["r"]
  s_ = ds["s_"]
  d = ds["d"]
  ds = TransitionTuple.new(obs, a, r.reshape(-1, 1), s_, d.reshape(-1, 1))

  rng = random.PRNGKey(cfg.sac.seed)
  obs, _ = env.reset(seed=cfg.sac.seed)

  pi, sample = sac.get_policy(mlp_conf, env.action_space.low,
                              env.action_space.high)
  critic = sac.get_critic(conf, mlp_conf)
  rng, params = rng_wrapper(sac.init_params)(rng, conf, mlp_conf, obs,
                                             env.action_space.high)
  train_state = sac.init_train(params, optax.adam(1e-3))
  update_fn = cql.make_train(
      conf,
      cql_conf,
      sample,
      critic,
      optax.adam(conf.actor_lr),
      optax.adam(conf.critic_lr),
      optax.adam(conf.actor_lr),
  )

  # Training
  step = 0
  with tqdm(desc="Training", total=cfg.cql.n_epochs, unit_scale=True) as pbar:
    for i in range(cfg.cql.n_epochs):
      rng, inds = rng_wrapper(random.permutation)(rng,
                                                  np.arange(len(ds.action)))
      n_batches = len(inds) // cfg.cql.batch_size
      inds = inds[:n_batches * cfg.cql.batch_size]
      shuffled = jax.tree_map(
          lambda x: x[inds].reshape(n_batches, cfg.cql.batch_size, -1), ds)

      def one_batch(update_state, minibatch):
        rng, params, train_state = update_state
        rng, (params, train_state,
              metrics) = rng_wrapper(update_fn)(rng, params, train_state,
                                                minibatch)
        return (rng, params, train_state), metrics

      (rng, params, train_state), metrics = lax.scan(one_batch,
                                                     (rng, params, train_state),
                                                     shuffled)
      for j in range(n_batches):
        logger.log(jax.tree_map(lambda x: x[j], metrics), step=step)
        step += 1

      if i % cfg.cql.eval_interval == 0:
        avg_rew = 0
        avg_steps = 0

        for j in range(cfg.cql.eval_n_trajs):
          rew = 0
          steps = 0

          eval_done = False
          eval_obs, _ = env.reset()
          while not eval_done:
            eval_a, _ = pi(params.actor, eval_obs)
            s_, r, d, d_, _ = env.step(eval_a)

            rew += r
            steps += 1
            eval_done = d or d_
            eval_obs = s_

          avg_rew += (rew - avg_rew) / (j + 1)
          avg_steps += (steps - avg_steps) / (j + 1)

        logger.log({"eval_reward": avg_rew}, step=step)
        pbar.set_postfix({"eval_reward": avg_rew, "eval_steps": avg_steps})

      pbar.update(1)


if __name__ == "__main__":
  main()
