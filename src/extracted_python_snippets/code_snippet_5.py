# id: newtype-covariance
# EXPECTED:
#   mypy: Error (invariance)
#   pyright: No error
#   pyre: Error (invariance)
#   zuban: Error (invariance)
# REASON: NewType container variance handling
from typing import NewType, List

AccountId = NewType('AccountId', int)
TransactionId = NewType('TransactionId', int)

def process_accounts(ids: List[AccountId]) -> None: ...

if __name__ == "__main__":
    accounts: List[AccountId] = [AccountId(1), AccountId(2)]
    numbers: List[int] = [1, 2, 3]
    
    process_accounts(accounts)  # Valid
    process_accounts(numbers)   # Checkers disagree on covariance