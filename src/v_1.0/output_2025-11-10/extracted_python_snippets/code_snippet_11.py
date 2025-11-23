# id: typeguard-tuple-narrow
# EXPECTED:
#   mypy: Item access error
#   pyright: No error
#   pyre: Item access error
#   zuban: No error
# REASON: TypeGuard narrowing for tuples handled differently
from typing import TypeGuard, Union

def is_int_tuple(val: tuple[Union[int, str], ...]) -> TypeGuard[tuple[int, ...]]:
    return all(isinstance(x, int) for x in val)

def process(data: tuple[Union[int, str], ...]) -> None:
    if is_int_tuple(data):
        data[0] + 1  # Checkers disagree on element type narrowing

if __name__ == "__main__":
    process((1, "two"))