from rflax.agents import sac
from rflax.utils import ReplayBuffer, TransitionTuple
from rflax.logging import WandBLogger

from jutils import np, random, rng_wrapper

import gymnasium as gym
import hydra
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg.sac))
  logger = WandBLogger(instantiate(cfg.sac.logging), OmegaConf.to_container(cfg.sac))

  env = gym.make(cfg.sac.env)
  conf = instantiate(cfg.sac.train)
  mlp_conf = instantiate(cfg.sac.mlp)

  rng = random.PRNGKey(cfg.sac.seed)
  obs, _ = env.reset(seed=cfg.sac.seed)

  pi, sample = sac.get_policy(mlp_conf, env.action_space.low, env.action_space.high)
  critic = sac.get_critic(conf, mlp_conf)
  rng, params = rng_wrapper(sac.init_params)(rng, conf, mlp_conf, obs, env.action_space.high)
  train_state = sac.init_train(params, optax.adam(1e-3))
  update_fn = sac.make_train(conf, sample, critic, optax.adam(conf.actor_lr), optax.adam(conf.critic_lr), optax.adam(conf.actor_lr))

  rb = ReplayBuffer(TransitionTuple.dummy(obs, env.action_space.low), cfg.sac.rb_size)

  # Training
  train = False
  done = False
  obs, _ = env.reset()
  with tqdm(desc="Training", total=cfg.sac.total_timesteps, unit_scale=True) as pbar:
    for i in range(cfg.sac.total_timesteps):
      rng, (a, _) = rng_wrapper(sample)(rng, params.actor, obs)
      next_obs, reward, terminated, truncated, _ = env.step(a)
      reward = np.array([reward])
      done = np.array([terminated or truncated])

      rb.add(TransitionTuple.new(obs, a, reward, next_obs, done))

      if train:
        rng, (params, train_state, metrics) = rng_wrapper(update_fn)(rng, params, train_state, rb.sample(rng, cfg.sac.batch_size))
        logger.log(metrics, step=i)

      if done:
        obs, _ = env.reset()
      else:
        obs = next_obs

      if not train and i >= cfg.sac.steps_before_train:
        train = True
        pbar.set_description("Training")
      if not train:
        pbar.set_description("Collecting")

      if train and i % cfg.sac.eval_interval == 0:
        avg_rew = 0
        avg_steps = 0

        for j in range(cfg.sac.eval_n_trajs):
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

        logger.log({"eval_reward": avg_rew}, step=i)
        pbar.set_postfix({"eval_reward": avg_rew, "eval_steps": avg_steps})

      pbar.update(1)


if __name__ == "__main__":
  main()
