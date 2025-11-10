# id: decorator-classmethod
# EXPECTED:
#   mypy: Error (signature mismatch)
#   pyright: No error
#   pyre: Error (signature mismatch)
#   zuban: Error (signature mismatch)
# REASON: Decorator signature preservation on classmethods
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

P = ParamSpec('P')
R = TypeVar('R')

def debug(func: Callable[P, R]) -> Callable[P, R]:
    return func

class API:
    @debug
    @classmethod
    def endpoint(cls, id: int) -> "API":  # Signature preservation
        return cls()

if __name__ == "__main__":
    API.endpoint(42)