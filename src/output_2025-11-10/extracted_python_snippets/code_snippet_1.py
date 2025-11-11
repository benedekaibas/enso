from typing import TypeGuard, TypeVar, Dict, Any
K = TypeVar('K')
V = TypeVar('V')

def is_str_dict(val: dict[Any, Any]) -> TypeGuard[dict[str, str]]:
    return all(isinstance(k, str) and isinstance(v, str) for k, v in val.items())

def process(data: Dict[object, object]) -> None:
    if is_str_dict(data):
        data['key'] = 'value'  # Checkers disagree on key/value type narrowing