from flax.core import FrozenDict
from rflax.agents.reinforce import REINFORCE
from rflax.utils import ReplayBuffer, to_jax_batch, squeeze_to_np, batch_to_jax
from rflax.logging import (
    define_flags_with_default,
    print_flags,
    WandBLogger,
    get_user_flags,
)

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags, logging

FLAGS_DEF = define_flags_with_default(
    env="Pendulum-v1",
    seed=42,
    total_timesteps=200000,
    eval_period=1000,
    eval_n_trajs=3,
    save_period=1000,
    ckpt_dir="ckpts/reinforce",
    load_ckpt=-1,
    reinforce=REINFORCE.get_default_config(),
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

  agent = REINFORCE(
      FLAGS.reinforce,
      rngs.pop(),
      env.observation_space.shape[0],
      (env.action_space.high, env.action_space.low),
  )

  if FLAGS.load_ckpt != -1:
    agent.restore_checkpoint(FLAGS.ckpt_dir, step=FLAGS.load_ckpt)

  # Training
  total_timesteps = 0
  steps_since_eval = 0
  steps_since_save = 0
  while total_timesteps < FLAGS.total_timesteps:
    done = False
    obs, _ = env.reset()

    episode_rew = 0
    episode_steps = 0
    states = []
    actions = []
    rewards = []
    while not done:
      a = squeeze_to_np(agent.sample_actions(to_jax_batch(obs)))
      next_obs, reward, done, mask, _ = env.step(a)

      states.append(obs)
      actions.append(a)
      rewards.append(reward)

      total_timesteps += 1
      steps_since_eval += 1
      steps_since_save += 1

      episode_rew += reward
      episode_steps += 1

      obs = next_obs
      done = done or mask

    batch = FrozenDict({
        "observations": jnp.array(states),
        "actions": jnp.array(actions),
        "rewards": jnp.array(rewards).reshape(-1, 1),
    })
    metrics = agent.update(batch)
    logging.info(f"| total_timesteps: {total_timesteps} "
                 f"| reward: {episode_rew:.3f} "
                 f"| steps: {episode_steps} "
                 f"| loss: {metrics['loss'].item()} |")
    metrics["episode_reward"] = episode_rew
    logger.log(metrics)

    if steps_since_eval > FLAGS.eval_period:
      steps_since_eval = 0
      avg_rew = 0
      avg_steps = 0

      for j in range(FLAGS.eval_n_trajs):
        rew = 0
        steps = 0

        eval_done = False
        eval_obs, _ = env.reset()
        while not eval_done:
          a = squeeze_to_np(agent.eval_actions(to_jax_batch(eval_obs)))
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

    if steps_since_save > FLAGS.save_period:
      steps_since_save = 0
      agent.save_checkpoint(
          FLAGS.ckpt_dir,
          keep=int(FLAGS.total_timesteps / FLAGS.save_period + 1),
          overwrite=True,
      )


if __name__ == "__main__":
  app.run(main)
