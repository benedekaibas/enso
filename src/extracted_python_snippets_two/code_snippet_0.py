# id: protocol-defaults-case1
# EXPECTED:
#   mypy: Accepts FileReader as Reader (default arg value differs is OK)
#   pyright: Reports error (default argument value doesn't match exactly)
#   pyre: Accepts FileReader as Reader (allows default arg differences)
#   zuban: Error (default arg clash)
# REASON: Not all type checkers agree whether default argument values must be equal for protocol compatibility.

from typing import Protocol

class Writer(Protocol):
    def write(self, data: bytes, flush: bool = True) -> int: ...

class FileWriter:
    def write(self, data: bytes, flush: bool = False) -> int:  # Different default for flush
        return len(data)

def use_writer(w: Writer) -> None:
    print(w.write(b"hi"))

if __name__ == "__main__":
    use_writer(FileWriter())  # Divergence on protocol implementation due to default args