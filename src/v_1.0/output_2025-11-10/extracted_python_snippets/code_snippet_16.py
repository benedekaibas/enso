# id: overload-literal-int
# EXPECTED:
#   mypy: Incompatible return type
#   pyright: Literal type preserved
#   pyre: Incompatible return type
#   zuban: Literal type preserved
# REASON: Literal type narrowing in overload resolution
from typing import overload, Literal, Union

@overload
def parse_int(value: Literal["0"]) -> Literal[0]: ...
@overload
def parse_int(value: str) -> Union[int, None]: ...

def parse_int(value: str) -> Union[int, None]:
    try: return int(value)
    except: return None

result = parse_int("0")  # Checkers disagree on Literal[0] vs int

if __name__ == "__main__":
    print(result)