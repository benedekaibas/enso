from typing import Final

class Base:
    @property
    def value(self) -> int:
        return 42

class Derived(Base):
    value: Final[int] = 100  # Checkers disagree on overriding property with attribute