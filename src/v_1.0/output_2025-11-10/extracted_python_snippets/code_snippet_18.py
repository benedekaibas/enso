# id: double-bound-generic
# EXPECTED:
#   mypy: Type argument error
#   pyright: No error
#   pyre: Type argument error
#   zuban: No error
# REASON: Nested generic bounds checking
from typing import TypeVar, Generic

T = TypeVar('T', bound=float)
U = TypeVar('U', bound=Generic[T])

class Container(Generic[T]): ...

def process(container: U) -> None: ...

if __name__ == "__main__":
    process(Container[float]())  # Checkers disagree on double-bound compatibility