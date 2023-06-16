"""Generic types used as pytype annotations."""

from typing import Callable, Sequence, Union, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.scope import FrozenVariableDict
from ml_collections import ConfigDict

Array = jax.Array
DType = jnp.dtype
PRNGKey = jax._src.prng.PRNGKeyArray

Shape = Sequence[int]

# Variable dict
VariableDict = Union[FrozenVariableDict, Dict[str, Any]]

# Config dict
ConfigDictLike = Union[ConfigDict, Dict[str, Any]]

# Metrics dict
MetricDict = Dict[str, Union[Array, float]]

# Data dict
DataDict = Dict[str, Union[np.ndarray, Array]]

# Paramter initializer
Initializer = Callable[[PRNGKey, Shape, DType], Array]

# Activation function (__init__ parameter)
ActivationFn = Callable[[Array], Array]
ActivationArg = Sequence[Union[str, ActivationFn]]
