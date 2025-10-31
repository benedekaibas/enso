# id: self-in-generics-case1
# EXPECTED:
#   mypy: Accepts Self, returns correct instance
#   pyright: Error, Self not allowed in generic ABC context
#   pyre: Error, Self in generics not supported
#   zuban: Error, Self cannot be used in generics
# REASON: `Self` is not universally supported in abstract generic classes, checkers disagree on specification usage.

from typing import Generic, TypeVar
from typing_extensions import Self
from abc import ABC, abstractmethod

T = TypeVar("T")

class Maker(ABC, Generic[T]):
    @abstractmethod
    def make(self) -> Self:
        ...

class TextMaker(Maker[str]):
    def make(self) -> Self:
        return self

if __name__ == "__main__":
    m = TextMaker()
    m2 = m.make()
    print(m is m2)