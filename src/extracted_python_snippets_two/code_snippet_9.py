# id: protocol-kwargs-positional-case1
# EXPECTED:
#   mypy: Accepts simple_func as KwargProtocol
#   pyright: Error, positional/keyword mismatch
#   pyre: Error, signature mismatch
#   zuban: Accepts, ignores positional vs keyword distinction
# REASON: Protocol matching is sensitive to call signatures (keyword-only), which is not consistently enforced by all checkers.

from typing import Protocol, Callable

class FuncProtocol(Protocol):
    def __call__(self, *, name: str, level: int) -> str: ...

def simple_func(name: str, level: int) -> str:  # Positional, not keyword-only params
    return f"{name}-{level}"

def run(handler: FuncProtocol) -> None:
    print(handler(name="Ann", level=3))

if __name__ == "__main__":
    run(simple_func)  # Should this be a type error?