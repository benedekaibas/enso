# id: typeguard-generic-case1
# EXPECTED:
#   mypy: Narrows data to list[int] inside block; append(int) is OK
#   pyright: Does NOT narrow data (TypeGuard loses info) 
#   pyre: Narrows data but loses generic; error on append
#   zuban: Fails to narrow list[object] to list[int]
# REASON: Some type checkers can't apply TypeGuard with generics, especially when narrowing container element types.

from typing import TypeGuard, TypeVar, List
T = TypeVar("T")

def is_list_of_type(xs: list[object], typ: type[T]) -> TypeGuard[list[T]]:
    return all(isinstance(x, typ) for x in xs)

def run(xs: List[object]) -> None:
    if is_list_of_type(xs, int):
        xs.append(3)  # Do checkers allow append(int) now?

if __name__ == "__main__":
    run([1, 2, 3])