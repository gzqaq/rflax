from rflax.agents.ddpg import DDPG
from rflax.utils import ReplayBuffer, to_jax_batch, squeeze_to_np, batch_to_jax
from rflax.logging import WandBLogger

import gymnasium as gym
import hydra
import jax
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg.ddpg))
  logger = WandBLogger(instantiate(cfg.ddpg.logging), OmegaConf.to_container(cfg.ddpg))

  env = gym.make(cfg.ddpg.env)

  rng = jax.random.PRNGKey(cfg.ddpg.seed)
  np.random.seed(cfg.ddpg.seed)
  obs, _ = env.reset(seed=cfg.ddpg.seed)

  rng, agent_rng = jax.random.split(rng)

  agent = DDPG(
      instantiate(cfg.ddpg.agent),
      agent_rng,
      env.observation_space.shape[0],
      (env.action_space.high, env.action_space.low),
  )
  rb = ReplayBuffer(cfg.ddpg.rb_size, agent.obs_dim, agent.action_dim)

  if cfg.ddpg.load_ckpt != -1:
    agent.restore_checkpoint(cfg.ddpg.ckpt_dir, step=cfg.ddpg.load_ckpt)

  # Training
  train = False
  done = False
  obs, _ = env.reset()
  with tqdm(desc="Training", total=cfg.ddpg.total_timesteps,
            unit_scale=True) as pbar:
    for i in range(cfg.ddpg.total_timesteps):
      a = squeeze_to_np(agent.sample_actions(to_jax_batch(obs)))
      next_obs, reward, done, mask, _ = env.step(a)

      reward = np.array(reward)
      done = np.array(done or mask)
      mask = 1 - done.astype(np.float32)

      rb.insert({
          "observations": obs,
          "actions": a,
          "rewards": reward,
          "next_observations": next_obs,
          "dones": done,
          "masks": mask,
      })

      if done:
        obs, _ = env.reset()
        if train:
          metrics = agent.update(batch_to_jax(rb.sample(cfg.ddpg.batch_size)))
          logger.log(metrics, step=i)
      else:
        obs = next_obs

      if not train and i >= cfg.ddpg.steps_before_train:
        train = True
        pbar.set_description("Training")
      if not train:
        pbar.set_description("Collecting")

      if train and i % cfg.ddpg.eval_interval == 0:
        avg_rew = 0
        avg_steps = 0

        for j in range(cfg.ddpg.eval_n_trajs):
          rew = 0
          steps = 0

          eval_done = False
          eval_obs, _ = env.reset()
          while not eval_done:
            eval_a = squeeze_to_np(agent.eval_actions(to_jax_batch(eval_obs)))
            s_, r, d, d_, _ = env.step(eval_a)

            rew += r
            steps += 1
            eval_done = d or d_
            eval_obs = s_

          avg_rew += (rew - avg_rew) / (j + 1)
          avg_steps += (steps - avg_steps) / (j + 1)

        logger.log({"eval_reward": avg_rew}, step=i)
        pbar.set_postfix({"eval_reward": avg_rew, "eval_steps": avg_steps})

      if train and i % cfg.ddpg.save_interval == 0:
        agent.save_checkpoint(
            cfg.ddpg.ckpt_dir,
            keep=int(cfg.ddpg.total_timesteps / cfg.ddpg.save_interval + 1),
            overwrite=True,
        )

      pbar.update(1)


if __name__ == "__main__":
  main()
