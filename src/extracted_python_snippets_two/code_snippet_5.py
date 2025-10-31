# id: newtype-list-covariance-case1
# EXPECTED:
#   mypy: error on process_user_ids(int_list) call
#   pyright: allows process_user_ids(int_list)
#   pyre: error, List[int] vs List[UserId] incompatible
#   zuban: allows List[int] as List[UserId] (covariant)
# REASON: NewType breaks nominal typing, but not all type checkers respect NewType as strictly as intended, especially in generics.

from typing import NewType, List

CustomerId = NewType('CustomerId', int)
OrderId = NewType('OrderId', int)

def handle_customers(ids: List[CustomerId]) -> None:
    for cid in ids:
        print(cid)

if __name__ == "__main__":
    handle_customers([CustomerId(7), CustomerId(8)])
    handle_customers([1, 2])  # Divergence: Should this be OK?