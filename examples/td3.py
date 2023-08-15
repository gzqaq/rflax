from rflax.agents.ddpg import TD3
from rflax.utils import ReplayBuffer, to_jax_batch, squeeze_to_np, batch_to_jax
from rflax.logging import (
    define_flags_with_default,
    print_flags,
    WandBLogger,
    get_user_flags,
)

import gymnasium as gym
import jax
import numpy as np
from absl import app, flags
from tqdm import tqdm

FLAGS_DEF = define_flags_with_default(
    env="Pendulum-v1",
    seed=42,
    rb_capacity=int(1e5),
    rb_minimal_size=1000,
    total_timesteps=1000000,
    batch_size=32,
    eval_period=1000,
    eval_n_trajs=3,
    save_period=10000,
    ckpt_dir="ckpts/td3",
    load_ckpt=-1,
    td3=TD3.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(_):
  FLAGS = flags.FLAGS
  print_flags(FLAGS, FLAGS_DEF)

  variant = get_user_flags(FLAGS, FLAGS_DEF)
  logger = WandBLogger(FLAGS.logging, variant)

  env = gym.make(FLAGS.env)

  rng = jax.random.PRNGKey(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  obs, _ = env.reset(seed=FLAGS.seed)

  rng, agent_rng = jax.random.split(rng)

  agent = TD3(
      FLAGS.td3,
      agent_rng,
      env.observation_space.shape[0],
      (env.action_space.high, env.action_space.low),
  )
  rb = ReplayBuffer(FLAGS.rb_capacity, agent.obs_dim, agent.action_dim)

  if FLAGS.load_ckpt != -1:
    agent.restore_checkpoint(FLAGS.ckpt_dir, step=FLAGS.load_ckpt)

  # Training
  train = False
  done = False
  obs, _ = env.reset()
  with tqdm(desc="Training", total=FLAGS.total_timesteps,
            unit_scale=True) as pbar:
    for i in range(FLAGS.total_timesteps):
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
          metrics = agent.update(batch_to_jax(rb.sample(FLAGS.batch_size)))
          logger.log(metrics, step=i)
      else:
        obs = next_obs

      if not train and rb.size >= FLAGS.rb_minimal_size:
        train = True
        pbar.set_description("Training")
      if not train:
        pbar.set_description("Collecting")

      if train and i % FLAGS.eval_period == 0:
        avg_rew = 0
        avg_steps = 0

        for j in range(FLAGS.eval_n_trajs):
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

      if train and i % FLAGS.save_period == 0:
        agent.save_checkpoint(
            FLAGS.ckpt_dir,
            keep=int(FLAGS.total_timesteps / FLAGS.save_period + 1),
            overwrite=True,
        )

      pbar.update(1)


if __name__ == "__main__":
  app.run(main)
