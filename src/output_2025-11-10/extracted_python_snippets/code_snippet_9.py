from typing import Protocol

class Handler(Protocol):
    def __call__(self, *, timeout: int) -> str: ...

def handle(timeout: int) -> str:
    return str(timeout)

handler: Handler = handle  # Checkers disagree on positional vs keyword