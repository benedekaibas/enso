from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, HttpUrl
import os
import httpx
import argparse
import json

class GetAccessToGitHubModels(BaseModel):
    """LLM based agent to send requests to GitHub models."""
    model: str = Field(..., description="Model id, e.g. 'openai/gpt-4.1'")
    api_base: HttpUrl = Field("https://models.github.ai", description="GitHub Models API base")
    timeout: float = Field(120.0, gt=0, description="Timeout (seconds)")
    token: str = Field(..., description="GitHub PAT with `models:read` scope")
    
    # Available models on GitHub Models
    AVAILABLE_MODELS: List[str] = [
        "openai/gpt-4.1",
        "openai/gpt-4",
        "openai/gpt-4-turbo", 
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "deepseek/DeepSeek-R1",
        "anthropic/claude-3-5-sonnet",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "mistralai/codestral-latest",
        "meta/llama-3-70b-instruct",
        "meta/llama-3-8b-instruct",
        "mistralai/mixtral-8x7b-instruct"
    ]

    def setup(
        self,
        model: Optional[str] = None, 
        api_base: Optional[HttpUrl] = None,
        timeout: Optional[float] = None,
        token: Optional[str] = None,
    ) -> None:
        """Validated updates (optional)."""
        updates: Dict[str, Any] = {}
        if model is not None:
            updates["model"] = model
        if api_base is not None: 
            updates["api_base"] = api_base
        if timeout is not None: 
            updates["timeout"] = timeout
        if token is not None: 
            updates["token"] = token
        if updates:
            new_self = self.model_copy(update=updates)
            self.model, self.api_base, self.timeout, self.token = (
                new_self.model, new_self.api_base, new_self.timeout, new_self.token
            )

    def communicate(self, prompt: str) -> str:
        """Send a prompt to GitHub Models and return the text reply."""
        base = str(self.api_base).rstrip('/')
        url = f"{base}/inference/chat/completions"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            choice0 = data.get("choices", [{}])[0]
            msg = (choice0.get("message") or {}).get("content")
            if not msg:
                raise ValueError("Invalid GitHub model response: missing choices[0].message.content")
            return str(msg)
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"HTTP {e.response.status_code} from {e.request.method} {e.request.url}: {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            raise ValueError(f"Network error contacting GitHub Models: {e}") from e

    def predict(self, prompt: str) -> str:
        return self.communicate(prompt)

    def print_models(self):
        """Display models the user can choose from in the terminal."""
        print("Available models on GitHub Models:")
        for i, model in enumerate(self.AVAILABLE_MODELS, 1):
            print(f"{i}. {model}")

    def cli_parser(self):
        """Creating a CLI to select different LLM models."""
        parser = argparse.ArgumentParser(description="Select the LLM model to use with GitHub Models")
        parser.add_argument(
            "--model", 
            choices=self.AVAILABLE_MODELS,
            default=self.model,
            help=f"Choose model from available options (default: {self.model})"
        )
        parser.add_argument(
            "--list-models",
            action="store_true",
            help="List all available models and exit"
        )
        return parser

# Much better prompt targeting real type checker divergences
EXPERT_PROMPT = """
You are an expert in Python type systems and type checker implementation differences. 
Generate 10 Python code snippets that demonstrate REAL, KNOWN divergences between 
mypy, pyright, pyre, and zuban type checkers.

CRITICAL REQUIREMENTS:
1. Each snippet must be SELF-CONTAINED and RUNNABLE
2. Use ONLY valid Python 3.11+ syntax and REAL typing features
3. NO forward reference issues - use string annotations if needed: `-> "Classname"`
4. Each snippet must target a SPECIFIC, DOCUMENTED type checker divergence
5. Include imports from `typing` and `typing_extensions` as needed
6. Include a minimal `if __name__ == "__main__":` block

TARGET THESE PROVEN DIVERGENCE AREAS:

1. **Protocol with default arguments**
```python
from typing import Protocol

class Reader(Protocol):
    def read(self, size: int = -1) -> bytes: ...

class FileReader:
    def read(self, size: int = 1024) -> bytes:  # Different default
        return b"data"

def use_reader(r: Reader) -> None: ...
use_reader(FileReader())  # Checkers disagree on default arg compatibility
```
```python
from typing import TypeGuard, TypeVar, List
T = TypeVar('T')

def is_list_of(val: list[object], type_: type[T]) -> TypeGuard[list[T]]:
    return all(isinstance(x, type_) for x in val)

def process(data: List[object]) -> None:
    if is_list_of(data, str):
        data.append("new")  # Checkers disagree on generic narrowing
```
```python
from typing import TypedDict
from typing_extensions import Required, NotRequired

class Base(TypedDict, total=False):
    x: int

class Child(Base):
    y: Required[str]  # Mixed total semantics
    z: NotRequired[int]

def test(td: Child) -> None:
    reveal_type(td.get('x'))  # Checkers disagree on optionality
```
```python
from typing import TypeVar, Callable
from typing_extensions import ParamSpec

P = ParamSpec('P')
T = TypeVar('T')

def log_calls(func: Callable[P, T]) -> Callable[P, T]:
    return func

class Service:
    @log_calls
    @classmethod
    def create(cls, name: str) -> "Service":  # Checkers disagree on signature preservation
        return cls()
```
```python
from typing import Generic, TypeVar
from typing_extensions import Self
from abc import ABC, abstractmethod

T = TypeVar('T')

class AbstractFactory(ABC, Generic[T]):
    @abstractmethod
    def create(self) -> Self:  # Checkers disagree on Self in generics
        ...

class ConcreteFactory(AbstractFactory[str]):
    def create(self) -> Self:
        return Self
```
```python
from typing import NewType, List

UserId = NewType('UserId', int)
ProductId = NewType('ProductId', int)

def process_user_ids(ids: List[UserId]) -> None: ...

user_ids: List[UserId] = [UserId(1), UserId(2)]
int_list: List[int] = [1, 2, 3]

process_user_ids(user_ids)  # All agree - ok
process_user_ids(int_list)   # Checkers disagree on List covariance
```
from typing import overload, Literal, Union

@overload
def parse(value: Literal["true"]) -> bool: ...
@overload  
def parse(value: Literal["false"]) -> bool: ...
@overload
def parse(value: str) -> str: ...

def parse(value: str) -> Union[bool, str]:
    if value == "true": return True
    if value == "false": return False  
    return value

result1 = parse("true")   # Checkers disagree on literal discrimination
result2 = parse("false")  # Some infer bool, some infer Union[bool, str]
```
```python
from typing import Final

class Base:
    x: Final[int] = 1

class Derived(Base):
    @property
    def x(self) -> int:  # Checkers disagree on Final override
        return 2
```
```python
from typing import TypeVar, Generic

class Animal: ...
class Dog(Animal): ...
class Cat(Animal): ...

T = TypeVar('T', bound=Animal)

class Container(Generic[T]): ...

class DogContainer(Container[Dog]):  # All agree - ok
    pass

# Checkers disagree on double-bound scenarios
U = TypeVar('U', bound=Container[Animal])
def process_container(cont: U) -> None: ...
```
```python
from typing import Protocol, Callable

class KwargProtocol(Protocol):
    def __call__(self, *, name: str, age: int) -> str: ...

def simple_func(name: str, age: int) -> str:
    return f"{name}_{age}"

handler: KwargProtocol = simple_func  # Checkers disagree on positional vs keyword
```

For EACH of the 10 snippets:

    Use the exact pattern above but make them unique

    Ensure they compile and run without syntax errors

    Include specific divergence expectations in comments

    Make the type issue clear but subtle

OUTPUT FORMAT for each:
    # id: <area>-<specific-case>
# EXPECTED:
#   mypy: <behavior>
#   pyright: <behavior> 
#   pyre: <behavior>
#   zuban: <behavior>
# REASON: <specific technical reason>
<complete runnable code>
Generate 10 such high-quality divergence examples.
"""

if __name__ == "__main__":
    token = os.environ.get("GITHUB_PAT")
    if not token:
        raise ValueError("Please set GITHUB_PAT environment variable")
    
    agent = GetAccessToGitHubModels(
        model="openai/gpt-4.1",
        token=token,
        api_base="https://models.github.ai",
        timeout=320.0,
    )
    
    # Handle command line arguments
    parser = agent.cli_parser()
    args = parser.parse_args()
    
    if args.list_models:
        agent.print_models()
        exit(0)
    
    if args.model:
        agent.setup(model=args.model)
    
    print(f"Using model: {agent.model}")
    print("Generating type checker divergence examples...")
    
    response = agent.predict(EXPERT_PROMPT)
    print("\n" + "="*60)
    print("GENERATED CODE EXAMPLES:")
    print("="*60)
    print(response)
"""
interactive model selection
python pydantic_better_version.py --interactive

model directly
python pydantic_better_version.py --model gpt-4-turbo --temperature 0.8
    """
