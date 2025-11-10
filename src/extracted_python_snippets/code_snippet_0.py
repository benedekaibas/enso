# id: protocol-default-args
# EXPECTED:
#   mypy: Error (default argument mismatch)
#   pyright: No error
#   pyre: Error (default argument mismatch)
#   zuban: Error (default argument mismatch)
# REASON: Protocol default argument strictness differs
from typing import Protocol

class Reader(Protocol):
    def read(self, size: int = -1) -> bytes: ...

class CustomReader:
    def read(self, size: int = 2048) -> bytes:
        return b"custom"

def use_reader(r: Reader) -> None:
    pass

if __name__ == "__main__":
    use_reader(CustomReader())  # Mismatched default value