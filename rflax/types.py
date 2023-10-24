"""Generic types for annotation."""

from jutils import (
    Array,
    DType,
    PRNGKey,
    Shape,
    VariableDict,
    DataDict,
    Initializer,
    ApplyFunction,
)

import optax
from typing import Callable, Any

SamplePolicy = Callable[[PRNGKey, VariableDict, Any], Any]

Optimizer = optax.GradientTransformation
OptState = optax.OptState
