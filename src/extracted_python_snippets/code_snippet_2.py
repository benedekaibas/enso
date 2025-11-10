# id: typeddict-mixed-total
# EXPECTED:
#   mypy: Optional[int]
#   pyright: int | None
#   pyre: int
#   zuban: int | None
# REASON: TypedDict total inheritance handling
from typing import TypedDict
from typing_extensions import NotRequired

class Base(TypedDict, total=False):
    x: int

class Child(Base, total=True):
    y: str
    x: NotRequired[int]  # Mixed total semantics

def test(td: Child) -> None:
    reveal_type(td.get('x'))  # Checkers disagree on optionality

if __name__ == "__main__":
    test({"y": "value"})