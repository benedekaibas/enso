# id: decorator-classmethod-return
# EXPECTED:
#   mypy: Incompatible return type
#   pyright: No error
#   pyre: Incompatible return type
#   zuban: No error
# REASON: Decorator chain preservation for classmethod returns
from typing import Callable, TypeVar, Any
from typing_extensions import ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def validate(func: Callable[P, T]) -> Callable[P, T]:
    return func

class Factory:
    @validate
    @classmethod
    def create(cls, value: int) -> "Factory":  # Checkers disagree on return type
        return cls()

if __name__ == "__main__":
    Factory.create(42)