# id: protocol-default-args-writer
# EXPECTED:
#   mypy: No error
#   pyright: Error
#   pyre: Error
#   zuban: No error
# REASON: Protocol parameter defaults aren't checked by all type checkers
from typing import Protocol

class Writer(Protocol):
    def write(self, data: bytes, mode: str = 'w') -> int: ...

class BufferedWriter:
    def write(self, data: bytes, mode: str = 'wb') -> int:  # Different default
        return len(data)

def use_writer(w: Writer) -> None: ...

if __name__ == "__main__":
    use_writer(BufferedWriter())  # Divergence in default arg compatibility