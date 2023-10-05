from rflax.agents.td3 import get_policy, get_critic, init_td3, init_train, make_train
from rflax.utils import ReplayBuffer, TransitionTuple
from rflax.logging import WandBLogger

import gymnasium as gym
import hydra
import jax
import numpy as np
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
  print(OmegaConf.to_yaml(cfg.td3))
  logger = WandBLogger(instantiate(cfg.td3.logging),
                       OmegaConf.to_container(cfg.td3))

  env = gym.make(cfg.td3.env)
  td3_conf = instantiate(cfg.td3.train)
  mlp_conf = instantiate(cfg.td3.mlp)

  rng = jax.random.PRNGKey(cfg.td3.seed)
  obs, _ = env.reset(seed=cfg.td3.seed)

  rng, _rng = jax.random.split(rng)
  eval_pi, sample_pi = get_policy(td3_conf, mlp_conf, env.action_space.low,
                                  env.action_space.high)
  critic = get_critic(td3_conf, mlp_conf)
  td3_state = init_td3(_rng, td3_conf, mlp_conf, obs, env.action_space.high)
  train_state = init_train(td3_state, optax.adam(td3_conf.actor_lr))
  update_fn = make_train(
      td3_conf,
      eval_pi,
      critic,
      optax.adam(td3_conf.actor_lr),
      optax.adam(td3_conf.critic_lr),
      env.action_space.low,
      env.action_space.high,
  )

  s_, r, d, *_ = env.step(env.action_space.low)
  r = np.array([r])
  d = np.array([d])
  rb = ReplayBuffer(
      TransitionTuple(obs=obs,
                      action=env.action_space.low,
                      reward=r,
                      next_obs=s_,
                      done=d),
      cfg.td3.rb_size,
  )

  # Training
  train = False
  done = False
  obs, _ = env.reset()
  with tqdm(desc="Training", total=cfg.td3.total_timesteps,
            unit_scale=True) as pbar:
    for i in range(cfg.td3.total_timesteps):
      rng, _rng = jax.random.split(rng)
      a = sample_pi(_rng, td3_state.actor_params, obs)
      next_obs, reward, terminated, truncated, _ = env.step(a)
      reward = np.array([reward])
      done = np.array([terminated or truncated])

      rb.add(
          TransitionTuple(obs=obs,
                          action=a,
                          reward=reward,
                          next_obs=next_obs,
                          done=done))

      if train:
        rng, _rng = jax.random.split(rng)
        td3_state, train_state, metrics = update_fn(
            _rng, td3_state, train_state, rb.sample(_rng, cfg.td3.batch_size))
        if metrics["actor_loss"] == 0:
          del metrics["actor_loss"]
        logger.log(metrics, step=i)

      if done:
        obs, _ = env.reset()
      else:
        obs = next_obs

      if not train and i >= cfg.td3.steps_before_train:
        train = True
        pbar.set_description("Training")
      if not train:
        pbar.set_description("Collecting")

      if train and i % cfg.td3.eval_interval == 0:
        avg_rew = 0
        avg_steps = 0

        for j in range(cfg.td3.eval_n_trajs):
          rew = 0
          steps = 0

          eval_done = False
          eval_obs, _ = env.reset()
          while not eval_done:
            eval_a = eval_pi(td3_state.actor_params, eval_obs)
            s_, r, d, d_, _ = env.step(eval_a)

            rew += r
            steps += 1
            eval_done = d or d_
            eval_obs = s_

          avg_rew += (rew - avg_rew) / (j + 1)
          avg_steps += (steps - avg_steps) / (j + 1)

        logger.log({"eval_reward": avg_rew}, step=i)
        pbar.set_postfix({"eval_reward": avg_rew, "eval_steps": avg_steps})

      # if train and i % cfg.td3.save_interval == 0:
      #   agent.save_checkpoint(
      #       cfg.td3.ckpt_dir,
      #       keep=int(cfg.td3.total_timesteps / cfg.td3.save_interval + 1),
      #       overwrite=True,
      #   )

      pbar.update(1)


if __name__ == "__main__":
  main()
