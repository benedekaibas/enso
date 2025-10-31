# id: typevar-double-bound-case1
# EXPECTED:
#   mypy: Allows double-bound TypeVar for generic container
#   pyright: Does NOT allow double-bound TypeVar
#   pyre: Error, double-bound TypeVar unsupported
#   zuban: Error, rejects compound bounds
# REASON: Multiple bounds (e.g. T bound to Container[Animal]) are implemented inconsistently in type checkers.

from typing import TypeVar, Generic

class Base: ...
class Sub(Base): ...
class Other(Base): ...

V = TypeVar('V', bound=Base)

class Thing(Generic[V]): ...

W = TypeVar('W', bound=Thing[Base])  # Compounded bound

def process(t: W) -> None:
    print(type(t))

if __name__ == "__main__":
    thing = Thing[Sub]()
    process(thing)