from rflax.components.nets.value import ActionValueEnsemble
from ..components.blocks import MlpConfig
from ..components.nets.policy import DetTanhPolicy
from ..components.noise import add_normal_noise
from ..types import Array, PRNGKey, SamplePolicy, VariableDict, ApplyFunction
from ..utils import get_apply_fn

from jutils import np, jit, rng_wrapper

from flax import struct
from typing import Tuple


@struct.dataclass
class TD3Config:
  n_qs: int = 2
  discount: float = 0.98
  tau: float = 0.005
  actor_lr: float = 1e-3
  critic_lr: float = 1e-3
  act_noise: float = 0.1
  target_noise: float = 0.2
  clip_noise: float = 0.5
  policy_delay: int = 2


@struct.dataclass
class TD3State:
  actor_params: VariableDict
  critic_params: VariableDict
  target_actor_params: VariableDict
  target_critic_params: VariableDict


def get_policy(action_dim: int, mlp_config: MlpConfig, td3_config: TD3Config, action_low: Array, action_high: Array) -> Tuple[ApplyFunction, SamplePolicy]:
  def policy(params: VariableDict, observation: Array) -> Array:
    a = get_apply_fn(DetTanhPolicy, action_dim, mlp_config)(params, observation)
    return np.where(a > 0, a * action_high, -a * action_low)

  def sample_policy(rng: PRNGKey, params: VariableDict, observation: Array) -> Array:
    a = policy(params, observation)
    return add_normal_noise(rng, a, 0, td3_config.act_noise).clip(action_low, action_high)

  return jit(policy), jit(sample_policy)


def init_td3(rng: PRNGKey, td3_conf: TD3Config, mlp_conf: MlpConfig, init_obs: Array, action_high: Array) -> TD3State:
  actor = DetTanhPolicy(action_high.shape[-1], mlp_conf, name="TD3_actor")
  critic = ActionValueEnsemble(td3_conf.n_qs, mlp_conf, name="TD3_critic")

  rng, (init_a, actor_params) = rng_wrapper(actor.init_with_output)(rng, init_obs)
  critic_params = critic.init(rng, init_obs, init_a)

  return TD3State(actor_params, critic_params, actor_params, critic_params)
