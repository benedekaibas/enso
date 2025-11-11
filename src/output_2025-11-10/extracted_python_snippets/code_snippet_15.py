# id: newtype-list-covariance
# EXPECTED:
#   mypy: List invariant error
#   pyright: No error
#   pyre: List invariant error
#   zuban: No error
# REASON: NewType list covariance handling differs
from typing import NewType, List

Handle = NewType('Handle', int)
Port = NewType('Port', int)

def open_connections(handles: List[Handle]) -> None: ...

if __name__ == "__main__":
    ports: List[Port] = [Port(80), Port(443)]
    open_connections(ports)  # Checkers disagree on NewType list compatibility