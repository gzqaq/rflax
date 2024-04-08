from typing import Any, TypeAlias

from flax.core.frozen_dict import FrozenDict

Variable: TypeAlias = FrozenDict[str, Any]
