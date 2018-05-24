from . import (doc_inherit, imported)

__all__ = sum(
    [m.__all__ for m in [doc_inherit, imported]],
    []
)

from .doc_inherit import *
from .imported import *
