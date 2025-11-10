# id: final-override
# EXPECTED:
#   mypy: Error (Final violation)
#   pyright: No error
#   pyre: Error (Final violation)
#   zuban: Error (Final violation)
# REASON: Final attribute override with property
from typing import Final

class Base:
    version: Final[int] = 1

class Sub(Base):
    @property
    def version(self) -> int:  # Override Final
        return 2

if __name__ == "__main__":
    print(Sub().version)