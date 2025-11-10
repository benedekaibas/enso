# id: overload-literal
# EXPECTED:
#   mypy: bool
#   pyright: bool
#   pyre: bool | str
#   zuban: bool
# REASON: Literal discrimination in overloads
from typing import overload, Literal, Union

@overload
def convert(val: Literal["on"]) -> bool: ...
@overload
def convert(val: Literal["off"]) -> bool: ...
@overload
def convert(val: str) -> str: ...

def convert(val: str) -> Union[bool, str]:
    if val == "on": return True
    if val == "off": return False
    return val

if __name__ == "__main__":
    reveal_type(convert("on"))  # Literal discrimination