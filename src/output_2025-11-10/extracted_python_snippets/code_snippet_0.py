from typing import Protocol

class Writer(Protocol):
    def write(self, data: bytes, mode: str = 'w') -> int: ...

class FileWriter:
    def write(self, data: bytes, mode: str = 'a') -> int:  # Different default 'a' vs 'w'
        return len(data)

def use_writer(w: Writer) -> None: ...
use_writer(FileWriter())  # Checkers disagree on default arg compatibility