from typing import Callable, TypeVar, Any
from typing_extensions import ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def debug(func: Callable[P, T]) -> Callable[P, T]:
    return func

class Logger:
    @debug
    @classmethod
    def log(cls, message: str) -> None:  # Checkers disagree on decorated classmethod type
        print(message)

Logger.log("test")  # Some checkers may complain about 'cls' argument