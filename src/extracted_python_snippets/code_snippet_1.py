# id: typeguard-narrowing
# EXPECTED:
#   mypy: Error (narrowing failure)
#   pyright: No error
#   pyre: Error (narrowing failure)
#   zuban: Error (narrowing failure)
# REASON: TypeGuard narrowing behavior in generics
from typing import TypeGuard, TypeVar, List

T = TypeVar('T')

def is_str_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(data: List[object]) -> None:
    if is_str_list(data):
        data.append(42)  # Should be invalid for list[str]

if __name__ == "__main__":
    process(["a", "b"])