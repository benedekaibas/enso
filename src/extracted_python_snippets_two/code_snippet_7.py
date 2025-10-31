# id: final-property-override-case1
# EXPECTED:
#   mypy: Error, cannot override Final attribute, including with @property
#   pyright: OK, property override allowed
#   pyre: Error, Final prevents property override
#   zuban: Allows property override of Final
# REASON: Not all checkers treat Final as blocking property overrides.

from typing import Final

class Parent:
    val: Final[int] = 100

class Child(Parent):
    @property
    def val(self) -> int:  # Should this be legal?
        return 200

if __name__ == "__main__":
    print(Child().val)