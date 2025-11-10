# id: self-generics
# EXPECTED:
#   mypy: Error (Self return)
#   pyright: No error
#   pyre: Error (Self return)
#   zuban: Error (Self return)
# REASON: Self type in generic abstract classes
from typing import Generic, TypeVar
from typing_extensions import Self
from abc import ABC, abstractmethod

T = TypeVar('T')

class Factory(ABC, Generic[T]):
    @abstractmethod
    def make(self) -> Self:  # Self in generics
        ...

class StringFactory(Factory[str]):
    def make(self) -> Self:
        return self

if __name__ == "__main__":
    StringFactory().make()