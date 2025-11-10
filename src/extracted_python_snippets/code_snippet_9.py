# id: callable-kwargs
# EXPECTED:
#   mypy: Error (positional vs keyword)
#   pyright: No error
#   pyre: Error (positional vs keyword)
#   zuban: Error (positional vs keyword)
# REASON: Callable protocol keyword argument matching
from typing import Protocol

class Handler(Protocol):
    def __call__(self, *, user: str, token: str) -> str: ...

def auth(user: str, token: str) -> str:
    return f"{user}:{token}"

if __name__ == "__main__":
    handler: Handler = auth  # Positional vs keyword mismatch