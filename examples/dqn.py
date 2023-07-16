from rflax.agents.dqn import DQN
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
from absl import app, flags, logging

FLAGS_DEF = define_flags_with_default(
    env="CartPole-v1",
    seed=42,
    rb_capacity=10000,
    rb_minimal_size=100,
    total_timesteps=200000,
    batch_size=32,
    eval_period=1000,
    eval_n_trajs=3,
    save_period=1000,
    ckpt_dir="ckpts/dqn",
    load_ckpt=-1,
    dqn=DQN.get_default_config(),
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

  rng, *rngs = jax.random.split(rng, 9)

  agent = DQN(FLAGS.dqn, rngs.pop(), env.observation_space.shape[0],
              env.action_space.n)
  rb = ReplayBuffer(FLAGS.rb_capacity, agent.obs_dim, 1, np.int32)

  if FLAGS.load_ckpt != -1:
    agent.restore_checkpoint(FLAGS.ckpt_dir, step=FLAGS.load_ckpt)

  # Training
  train = False
  total_timesteps = 0
  steps_since_eval = 0
  steps_since_save = 0
  while total_timesteps < FLAGS.total_timesteps:
    done = False
    obs, _ = env.reset()

    episode_rew = 0
    episode_steps = 0
    while not done:
      a = squeeze_to_np(agent.sample_actions(to_jax_batch(obs))).item()
      next_obs, reward, done, mask, _ = env.step(a)

      total_timesteps += 1
      steps_since_eval += 1
      steps_since_save += 1

      episode_rew += reward
      episode_steps += 1

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

      obs = next_obs

    if rb.size >= FLAGS.rb_minimal_size:
      train = True
      metrics = agent.update(batch_to_jax(rb.sample(FLAGS.batch_size)))
      logging.info(f"| total_timesteps: {total_timesteps} "
                   f"| reward: {episode_rew:.3f} "
                   f"| steps: {episode_steps} "
                   f"| q_loss: {metrics['q_loss'].item()} |")
      metrics["episode_reward"] = episode_rew
      logger.log(metrics)
    else:
      logging.info(f"| Collecting training data: {total_timesteps} steps |")

    if train and steps_since_eval > FLAGS.eval_period:
      steps_since_eval = 0
      avg_rew = 0
      avg_steps = 0

      for j in range(FLAGS.eval_n_trajs):
        rew = 0
        steps = 0

        eval_done = False
        eval_obs, _ = env.reset()
        while not eval_done:
          a = squeeze_to_np(agent.eval_actions(to_jax_batch(eval_obs))).item()
          eval_obs, eval_r, d, d_, _ = env.step(a)

          rew += eval_r
          steps += 1
          eval_done = d or d_

        avg_rew += (rew - avg_rew) / (j + 1)
        avg_steps += (steps - avg_steps) / (j + 1)

      logging.info(f"| Evaluation "
                   f"| avg_reward: {avg_rew:.3f} "
                   f"| avg_steps: {avg_steps} |")
      logger.log({"eval_reward": avg_rew})

    if train and steps_since_save > FLAGS.save_period:
      steps_since_save = 0
      agent.save_checkpoint(
          FLAGS.ckpt_dir,
          keep=int(FLAGS.total_timesteps / FLAGS.save_period + 1),
          overwrite=True,
      )


if __name__ == "__main__":
  app.run(main)
