# id: paramspec-classmethod-decorator-case1
# EXPECTED:
#   mypy: Accepts, signature preserved, no error
#   pyright: Reports error, decorator breaks @classmethod signature
#   pyre: Accepts (but sometimes ignores decorator stack), no error
#   zuban: Error, cannot compose ParamSpec and @classmethod
# REASON: Not all type checkers understand stacked decorators (ParamSpec then @classmethod) and can infer proper classmethod signature.

from typing import Callable, TypeVar
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def passthru(f: Callable[P, R]) -> Callable[P, R]:
    return f

class Build:
    @passthru
    @classmethod
    def construct(cls, thing: str) -> "Build":
        return cls()

if __name__ == "__main__":
    print(Build.construct("item"))