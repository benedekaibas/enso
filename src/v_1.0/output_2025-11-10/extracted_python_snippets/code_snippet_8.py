from typing import TypeVar, Generic

class Box(Generic[T]):
    pass

T2 = TypeVar('T2', bound=Box[float])

def unpack(box: T2) -> None: ...

float_box: Box[float] = Box()
unpack(float_box)  # Checkers disagree if T2 bound is properly enforced