# id: typeddict-mixed-required
# EXPECTED:
#   mypy: 'name' is required
#   pyright: 'name' is optional
#   pyre: 'name' is required
#   zuban: 'name' is optional
# REASON: Total inheritance semantics differ across checkers
from typing import TypedDict
from typing_extensions import Required, NotRequired

class Base(TypedDict, total=True):
    id: int

class User(Base, total=False):
    name: Required[str]
    age: NotRequired[int]

def handle_user(u: User) -> None:
    u['id']  # All agree required
    u['name']  # Checkers disagree on required status

if __name__ == "__main__":
    handle_user({'id': 1, 'name': 'Alice'})