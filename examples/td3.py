from rflax.agents import td3
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
  print(OmegaConf.to_yaml(cfg.td3))
  logger = WandBLogger(instantiate(cfg.td3.logging),
                       OmegaConf.to_container(cfg.td3))

  env = gym.make(cfg.td3.env)
  conf = instantiate(cfg.td3.train)
  mlp_conf = instantiate(cfg.td3.mlp)

  rng = random.PRNGKey(cfg.td3.seed)
  obs, _ = env.reset(seed=cfg.td3.seed)

  eval_pi, sample_pi = td3.get_policy(conf, mlp_conf, env.action_space.low, env.action_space.high)
  critic = td3.get_critic(conf, mlp_conf)
  rng, params = rng_wrapper(td3.init_params)(rng, conf, mlp_conf, obs, env.action_space.high)
  train_state = td3.init_train(params, optax.adam(1e-3))
  update_fn = td3.make_train(conf, eval_pi, critic, optax.adam(conf.actor_lr), optax.adam(conf.critic_lr), env.action_space.low, env.action_space.high)

  rb = ReplayBuffer(TransitionTuple.dummy(obs, env.action_space.low), cfg.td3.rb_size)

  # Training
  train = False
  done = False
  obs, _ = env.reset()
  with tqdm(desc="Training", total=cfg.td3.total_timesteps,
            unit_scale=True) as pbar:
    for i in range(cfg.td3.total_timesteps):
      rng, a = rng_wrapper(sample_pi)(rng, params.actor, obs)
      next_obs, reward, terminated, truncated, _ = env.step(a)
      reward = np.array([reward])
      done = np.array([terminated or truncated])

      rb.add(TransitionTuple.new(obs, a, reward, next_obs, done))

      if train:
        rng, (params, train_state, metrics) = rng_wrapper(update_fn)(rng, params, train_state, rb.sample(rng, cfg.td3.batch_size))
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
            eval_a = eval_pi(params.actor, eval_obs)
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
