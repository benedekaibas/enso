# id: typeddict-mixed-total-case1
# EXPECTED:
#   mypy: child.get('foo') is Optional with total=False
#   pyright: child.get('foo') as str, not Optional
#   pyre: Treats as required, not Optional
#   zuban: Required/NotRequired mixing not supported, error or wrong optionality
# REASON: Most type checkers disagree on how `Required` and `NotRequired` interact inside a partially total TypedDict.

from typing import TypedDict
from typing_extensions import Required, NotRequired

class TDBase(TypedDict, total=False):
    foo: int

class TDChild(TDBase):
    bar: Required[str]
    baz: NotRequired[float]

def show(td: TDChild) -> None:
    print(td.get('foo'))  # Is this Optional[int], int, or error?

if __name__ == "__main__":
    show({'bar': 'x', 'foo': 42})