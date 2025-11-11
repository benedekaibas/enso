from typing import NewType, List

AccountID = NewType('AccountID', int)
TransactionID = NewType('TransactionID', int)

def process_accounts(ids: List[AccountID]) -> None: ...

mixed_list: List[int] = [AccountID(100), 200]  # Contains both AccountID and int
process_accounts(mixed_list)  # Checkers disagree on List[AccountID] vs List[int]