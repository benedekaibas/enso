# id: protocol-keyword-args
# EXPECTED:
#   mypy: Incompatible types
#   pyright: No error
#   pyre: Incompatible types
#   zuban: No error
# REASON: Positional vs keyword argument protocol matching
from typing import Protocol

class Formatter(Protocol):
    def format(self, *, text: str, width: int) -> str: ...

def simple_format(text: str, width: int) -> str:
    return f"{text:{width}}"

handler: Formatter = simple_format  # Checkers disagree on positional params

if __name__ == "__main__":
    handler.format(text="test", width=10)