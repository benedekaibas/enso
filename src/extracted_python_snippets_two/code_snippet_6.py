# id: overload-literal-discrim-case1
# EXPECTED:
#   mypy: result1 inferred as bool, result2 as bool
#   pyright: result1 and result2 as Union[bool, str]
#   pyre: result1 Union[bool, str], result2 Union[bool, str]
#   zuban: result1 as bool, result2 as Union[bool, str]
# REASON: Literal narrowing in overloads is inconsistently applied across type checkers.

from typing import overload, Literal, Union

@overload
def flag(v: Literal["yes"]) -> bool: ...
@overload
def flag(v: Literal["no"]) -> bool: ...
@overload
def flag(v: str) -> str: ...

def flag(v: str) -> Union[bool, str]:
    if v == "yes": return True
    if v == "no": return False
    return v

resp1 = flag("yes")
resp2 = flag("no")

if __name__ == "__main__":
    print(resp1, resp2)