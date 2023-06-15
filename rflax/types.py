"""Generic types used as pytype annotations."""

from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp

Array = jax.Array
DType = jnp.dtype
PRNGKey = jax._src.prng.PRNGKeyArray

Shape = Sequence[int]

# Paramter initializer
Initializer = Callable[[PRNGKey, Shape, DType], Array]

# Activation function (__init__ parameter)
ActivationFn = Callable[[Array], Array]
ActivationArg = Sequence[Union[str, ActivationFn]]
