from typing import Generic, TypeVar
from typing_extensions import Self
from abc import ABC, abstractmethod

T = TypeVar('T')

class Builder(ABC, Generic[T]):
    @abstractmethod
    def build(self) -> Self: ...

class IntBuilder(Builder[int]):
    def build(self) -> Self:
        return self  # Checkers disagree on return type compatibility