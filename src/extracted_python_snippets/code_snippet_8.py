# id: double-bound-generics
# EXPECTED:
#   mypy: Error (type bound)
#   pyright: No error
#   pyre: Error (type bound)
#   zuban: Error (type bound)
# REASON: Double-bound TypeVar with generics
from typing import TypeVar, Generic

class Animal: pass
class Mammal(Animal): pass

T = TypeVar('T', bound=Animal)

class Cage(Generic[T]): pass

U = TypeVar('U', bound=Cage[Animal])

def examine(cage: U) -> None: ...

if __name__ == "__main__":
    examine(Cage[Mammal]())  # Checkers disagree on bound compatibility