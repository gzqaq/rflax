import flax.linen as nn
import jax
import jax.numpy as np

from rflax.common.dist import TanhNormal


class ActorCritic(nn.Module):
  action_dim: int
  log_std_init: float = 0.0
  hidden_dim: int = 64
  act_fn: str = "tanh"

  @nn.compact
  def __call__(self, s: jax.Array) -> tuple[TanhNormal, jax.Array]:
    act_fn = getattr(nn, self.act_fn)

    o_init = nn.initializers.orthogonal
    hidden_init = o_init(np.sqrt(2))
    v_o_init = o_init(1)
    a_o_init = o_init(0.01)

    a_mean = act_fn(nn.Dense(self.hidden_dim, kernel_init=hidden_init)(s))
    a_mean = act_fn(nn.Dense(self.hidden_dim, kernel_init=hidden_init)(a_mean))
    a_mean = nn.Dense(self.action_dim, kernel_init=a_o_init)(a_mean)
    log_std = self.param("log_std", nn.initializers.constant(self.log_std_init),
                         (self.action_dim,))
    pi = TanhNormal(a_mean, np.exp(log_std))

    critic = act_fn(nn.Dense(self.hidden_dim, kernel_init=hidden_init)(s))
    critic = act_fn(nn.Dense(self.hidden_dim, kernel_init=hidden_init)(critic))
    critic = nn.Dense(1, kernel_init=v_o_init)(critic)

    return pi, critic.squeeze(-1)
