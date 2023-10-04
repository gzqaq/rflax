"""Generic types for annotation."""

from jutils import Array, DType, PRNGKey, Shape, VariableDict, DataDict, Initializer, ApplyFunction

from typing import Callable, Any


SamplePolicy = Callable[[PRNGKey, VariableDict, Any], Any]
