from rflax.agents.ddpg import DDPG
from rflax.utils import ReplayBuffer, to_jax_batch, squeeze_to_np, batch_to_jax
from rflax.logging import define_flags_with_default, print_flags

import gymnasium as gym
import jax
import numpy as np
from absl import app, flags, logging

FLAGS_DEF = define_flags_with_default(
    env="Pendulum-v1",
    seed=42,
    rb_capacity=int(1e5),
    rb_minimal_size=1000,
    n_epochs=10000,
    batch_size=512,
    eval_period=5,
    eval_n_trajs=3,
    save_period=100,
    ckpt_dir="ckpts/ddpg",
    load_ckpt=-1,
    ddpg=DDPG.get_default_config(),
)


def main(_):
  FLAGS = flags.FLAGS
  print_flags(FLAGS, FLAGS_DEF)

  env = gym.make(FLAGS.env)

  rng = jax.random.PRNGKey(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  obs, _ = env.reset(seed=FLAGS.seed)

  rng, *rngs = jax.random.split(rng, 9)

  agent = DDPG(
      FLAGS.ddpg,
      rngs.pop(),
      env.observation_space.shape[0],
      (env.action_space.high, env.action_space.low),
  )
  rb = ReplayBuffer(FLAGS.rb_capacity, agent.obs_dim, agent.action_dim)

  if FLAGS.load_ckpt != -1:
    agent.restore_checkpoint(FLAGS.ckpt_dir, step=FLAGS.load_ckpt)

  # Training
  train = False
  for i in range(FLAGS.n_epochs):
    done = False
    obs, _ = env.reset()

    episode_rew = 0
    episode_steps = 0
    while not done:
      a = squeeze_to_np(agent.sample_actions(to_jax_batch(obs)))
      next_obs, reward, done, mask, _ = env.step(a)

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
      logging.info(f"| Epoch {i} "
                   f"| reward: {episode_rew:.3f} "
                   f"| steps: {episode_steps} "
                   f"| actor_loss: {metrics['actor_loss'].item()} "
                   f"| critic_loss: {metrics['critic_loss'].item()} |")
    else:
      logging.info(f"| Epoch {i} | collecting training data... |\r")

    if train and i % FLAGS.eval_period == 0:
      avg_rew = 0
      avg_steps = 0

      for j in range(FLAGS.eval_n_trajs):
        rew = 0
        steps = 0

        done = False
        obs, _ = env.reset()
        while not done:
          a = squeeze_to_np(agent.eval_actions(to_jax_batch(obs)))
          s_, r, d, d_, _ = env.step(a)

          rew += r
          steps += 1
          done = d or d_
          obs = s_

        avg_rew += (rew - avg_rew) / (j + 1)
        avg_steps += (steps - avg_steps) / (j + 1)

      logging.info(f"| Evaluation "
                   f"| avg_reward: {avg_rew:.3f} "
                   f"| avg_steps: {avg_steps} |")

    if train and i % FLAGS.save_period == 0:
      agent.save_checkpoint(
          FLAGS.ckpt_dir,
          keep=int(FLAGS.n_epochs / FLAGS.save_period + 1),
          overwrite=True,
      )


if __name__ == "__main__":
  app.run(main)
