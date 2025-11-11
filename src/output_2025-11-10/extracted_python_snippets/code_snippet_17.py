# id: final-method-override
# EXPECTED:
#   mypy: Cannot override final method
#   pyright: No error
#   pyre: Cannot override final method
#   zuban: No error
# REASON: Final method override checking strictness
from typing import final

class Base:
    @final
    def critical(self) -> None: ...

class Derived(Base):
    def critical(self) -> None:  # Checkers disagree on final enforcement
        print("Overridden")

if __name__ == "__main__":
    Derived().critical()