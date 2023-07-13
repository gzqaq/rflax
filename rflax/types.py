"""Generic types used as pytype annotations."""

from typing import Callable, Union, Dict, Any, Mapping

import chex
import jax
import numpy as np
from flax.core.scope import FrozenVariableDict
from ml_collections import ConfigDict

Array = jax.Array
DType = chex.ArrayDType
PRNGKey = chex.PRNGKey

Shape = chex.Shape

# Variable dict
VariableDict = Union[FrozenVariableDict, Dict[str, Any]]

# Config dict
ConfigDictLike = Union[ConfigDict, Dict[str, Any], Mapping[str, Any]]

# Metrics dict
MetricDict = Dict[str, Union[Array, float]]

# Data dict
DataDict = Dict[str, Union[np.ndarray, Array]]

# Paramter initializer
Initializer = Callable[[PRNGKey, Shape, DType], Array]

# Activation function (__init__ parameter)
ActivationFn = Callable[[Array], Array]
