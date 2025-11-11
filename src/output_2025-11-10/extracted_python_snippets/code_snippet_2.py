from typing import TypedDict
from typing_extensions import NotRequired

class Parent(TypedDict, total=True):
    a: int
    b: NotRequired[str]

class Child(Parent, total=False):
    c: float  # Now 'a' is required, 'b' is not, 'c' is not required

def test(td: Child) -> None:
    td['a']  # Checkers disagree on required status of 'a' in Child