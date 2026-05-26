"""Block sparsity types — dense / spec build only; sparse paths are not supported here."""

from typing import Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BlockSparseTensors:
    """Placeholder for forward/sparse; dense backward always passes None."""

    _dummy: Any = None
    block_size: Optional[Tuple[int, int]] = None
