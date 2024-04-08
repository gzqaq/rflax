import chex as cx
import distrax as dx
import jax


class TanhNormal(dx.Transformed):

  def __init__(self, loc: jax.Array, scale_diag: jax.Array) -> None:
    mvn = dx.MultivariateNormalDiag(loc, scale_diag)
    tanh_bijector = dx.Block(dx.Tanh(), 1)

    super().__init__(distribution=mvn, bijector=tanh_bijector)

  def mean(self) -> jax.Array:
    return self.bijector.forward(self.distribution.mean())

  def entropy(self, input_hint: cx.Array | None = None) -> cx.Array:
    return self.distribution.entropy()
