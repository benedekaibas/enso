# id: self-generic-inheritance
# EXPECTED:
#   mypy: Return type incompatible
#   pyright: No error
#   pyre: Incompatible return type
#   zuban: No error
# REASON: Self type in generic base class handling
from typing import Generic, TypeVar
from typing_extensions import Self
from abc import ABC, abstractmethod

T = TypeVar('T')

class Template(ABC, Generic[T]):
    @abstractmethod
    def clone(self) -> Self: ...

class StringTemplate(Template[str]):
    def clone(self) -> Self:
        return self  # Checkers disagree on Self compatibility

if __name__ == "__main__":
    StringTemplate().clone()