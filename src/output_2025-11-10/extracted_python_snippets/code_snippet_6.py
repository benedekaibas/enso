from typing import overload, Literal, Union

@overload
def convert(val: Literal[1]) -> str: ...
@overload
def convert(val: Literal[2]) -> int: ...
@overload
def convert(val: int) -> float: ...

def convert(val: int) -> Union[str, int, float]:
    if val == 1:
        return "one"
    elif val == 2:
        return 2
    return float(val)

result = convert(1)  # Checkers disagree: str vs Union[...]