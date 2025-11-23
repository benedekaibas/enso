"""
Type Checker False Positive/Negative Discovery Pipeline

Extract code examples from closed GitHub issues (2023-2025) and tweak them to discover
new false positive and false negative cases in type checkers.

Type Checkers:
- mypy (python/mypy)
- ty (astral-sh/ty)
- pyrefly (facebook/pyrefly)
- zuban (zubanls/zuban)

Pipeline:
1. Fetch closed issues with code from type checker repos (2023-2025)
2. Extract original code directly from issue
3. Tweak code (LLM) - modify the original to trigger FP/FN
4. Run type checkers on tweaked code
5. If divergence found -> success, save result
6. If no divergence -> tweak again, repeat until divergence or max rounds

Usage:
    python pipeline.py --list-models
    python pipeline.py --model openai/gpt-4o --num-examples 5
    python pipeline.py --model meta/Llama-3.3-70B-Instruct --num-examples 3 --max-rounds 15
"""

from typing import Dict, Optional, List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider
import os
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import re
import httpx
import asyncio
import random

# ============================================================================
# Available GitHub Models
# ============================================================================

AVAILABLE_MODELS: List[str] = [
    "openai/gpt-4.1",
    "openai/gpt-4",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "deepseek/DeepSeek-R1",
    "deepseek/DeepSeek-R1-0528",
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "mistralai/codestral-latest",
    "meta/Llama-3.3-70B-Instruct",
    "meta/llama-3-8b-instruct",
    "mistralai/mixtral-8x7b-instruct",
]

# Type checker repositories and their issue trackers
TYPE_CHECKER_REPOS = {
    "python/mypy": "https://github.com/python/mypy/issues",
    "astral-sh/ty": "https://github.com/astral-sh/ty/issues",
    "facebook/pyrefly": "https://github.com/facebook/pyrefly/issues",
    "zubanls/zuban": "https://github.com/zubanls/zuban/issues",
}

# Date range for issues (2023-2025)
ISSUE_DATE_RANGE = "2023-01-01"


def print_available_models() -> None:
    print("\nAvailable GitHub Models:")
    print("-" * 40)
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i:2}. {model}")
    print()


def validate_model(model: str) -> bool:
    return model in AVAILABLE_MODELS


# ============================================================================
# Data Models
# ============================================================================

class ExtractedIssue(BaseModel):
    """A closed issue with extracted code example"""
    repo: str
    issue_number: int
    issue_url: str              # Direct link to this specific issue
    issue_tracker_url: str      # Link to the repo's issue tracker
    title: str
    original_code: str          # The actual code from the issue - NOT modified
    issue_description: str
    labels: List[str] = []


class TweakedExample(BaseModel):
    """A tweaked code example designed to trigger false pos/neg"""
    id: str
    source_issue_url: str        # Direct link to the specific issue
    source_issue_tracker: str    # Link to the repo's issue tracker
    original_code: str           # Original code from issue
    tweaked_code: str            # LLM-tweaked version
    tweak_description: str       # What the LLM changed


class ValidationResult(BaseModel):
    """Result of running type checkers on a tweaked example"""
    example_id: str
    checker_outputs: Dict[str, str]
    checker_classifications: Dict[str, str]  # ERROR, CLEAN, WARNING per checker
    has_divergence: bool
    is_valid_syntax: bool
    analysis: str


# ============================================================================
# Stage 1 & 2: Issue Fetcher - Get closed issues and extract code
# ============================================================================

class IssueFetcher:
    """Fetches closed issues from type checker repositories and extracts code"""

    def __init__(self, github_token: str):
        self.github_token = github_token
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    async def fetch_closed_issues(self, repo: str, max_issues: int = 50) -> List[Dict]:
        """Fetch closed issues from a repository that contain code examples (2023-2025)"""
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "state": "closed",
            "per_page": max_issues,
            "sort": "updated",
            "direction": "desc",
            "since": ISSUE_DATE_RANGE,  # Only issues from 2023 onwards
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, headers=self.headers, params=params, timeout=30.0
                )
                response.raise_for_status()
                issues = response.json()
                
                # Filter for issues that have code blocks
                code_issues = [
                    issue for issue in issues
                    if issue.get("body") and "```" in issue.get("body", "")
                ]
                return code_issues
        except Exception as e:
            print(f"      Error fetching from {repo}: {e}")
            return []

    async def fetch_issues_with_code(self, num_issues: int = 10) -> List[ExtractedIssue]:
        """Fetch closed issues with code from type checker repos"""
        all_issues = []

        for repo, issue_tracker_url in TYPE_CHECKER_REPOS.items():
            issues = await self.fetch_closed_issues(repo, max_issues=100)
            
            for issue in issues:
                code_blocks = self._extract_code_blocks(issue.get("body", ""))
                if code_blocks:
                    code = self._select_best_code_block(code_blocks)
                    if code and len(code) > 20:
                        all_issues.append(ExtractedIssue(
                            repo=repo,
                            issue_number=issue["number"],
                            issue_url=issue["html_url"],
                            issue_tracker_url=issue_tracker_url,
                            title=issue["title"],
                            original_code=code,
                            issue_description=issue.get("body", "")[:1000],
                            labels=[l["name"] for l in issue.get("labels", [])],
                        ))

        # Randomly select the requested number
        if len(all_issues) > num_issues:
            selected = random.sample(all_issues, num_issues)
        else:
            selected = all_issues

        print(f"Found {len(selected)} issues with code")
        return selected

    def _extract_code_blocks(self, body: str) -> List[str]:
        """Extract Python code blocks from issue body"""
        if not body:
            return []
        
        pattern = r"```(?:python|py)?\s*\n(.*?)```"
        matches = re.findall(pattern, body, re.DOTALL | re.IGNORECASE)
        
        python_blocks = []
        for block in matches:
            block = block.strip()
            if any(kw in block for kw in ["def ", "class ", "import ", "from ", ":", "->", "Type"]):
                python_blocks.append(block)
        
        return python_blocks

    def _select_best_code_block(self, blocks: List[str]) -> Optional[str]:
        """Select the most relevant code block (prefer ones with type hints)"""
        if not blocks:
            return None
        
        for block in blocks:
            if any(hint in block for hint in ["->", ": int", ": str", ": List", ": Dict", "Optional", "Union", "TypeVar"]):
                return block
        
        return max(blocks, key=len)


# ============================================================================
# Known Divergence Examples - Patterns that cause type checkers to disagree
# ============================================================================

# ============================================================================
# Known Divergence Examples - Patterns that cause type checkers to disagree
# ============================================================================

DIVERGENCE_EXAMPLES = """
## KNOWN DIVERGENCE PATTERNS

These are REAL patterns where type checkers DISAGREE. Use these as templates:

### 1. Callable vs Protocol structural matching
```python
from typing import Protocol, Callable

class MyProtocol(Protocol):
    def __call__(self, x: int) -> int: ...

def takes_protocol(f: MyProtocol) -> int:
    return f(1)

def my_func(x: int) -> int:
    return x * 2

# Some checkers accept functions as Protocol implementers, others don't
takes_protocol(my_func)
```

### 2. Type narrowing in nested functions
```python
def outer(x: str | None) -> str:
    def inner() -> str:
        if x is None:
            return ""
        # Does inner() see the narrowed type of x?
        return x.upper()
    return inner()
```

### 3. ClassVar in dataclass
```python
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class Config:
    name: str
    MAX_SIZE: ClassVar[int] = 100

# Some checkers error on ClassVar usage in dataclasses
c = Config("test")
print(Config.MAX_SIZE)
```

### 4. Mutable default in TypedDict
```python
from typing import TypedDict, Required

class MyDict(TypedDict, total=False):
    name: Required[str]
    tags: list[str]

def make_dict() -> MyDict:
    return {"name": "test"}  # Is missing 'tags' ok?
```

### 5. Generic Self in classmethod
```python
from typing import Self, TypeVar, Generic

T = TypeVar("T")

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
    
    @classmethod
    def create(cls, value: T) -> Self:
        return cls(value)

# Self in classmethod with generics
c: Container[int] = Container.create(42)
```

### 6. Intersection of Protocols
```python
from typing import Protocol

class HasName(Protocol):
    name: str

class HasAge(Protocol):
    age: int

def needs_both(obj: HasName & HasAge) -> str:  # type: ignore
    return f"{obj.name}: {obj.age}"
```

### 7. Callback with default arguments
```python
from typing import Callable

def decorator(func: Callable[[int], int]) -> Callable[[int], int]:
    return func

@decorator
def my_func(x: int = 0) -> int:  # Default arg - does it still match?
    return x + 1

result = my_func()
```

### 8. Type alias with forward reference
```python
from typing import TypeAlias

MyList: TypeAlias = list["MyClass"]

class MyClass:
    def get_related(self) -> MyList:
        return []
```

### 9. Abstract property override
```python
from abc import ABC, abstractmethod

class Base(ABC):
    @property
    @abstractmethod
    def value(self) -> int: ...

class Derived(Base):
    value: int = 10  # Property -> attribute, is this ok?
```

### 10. Unpack with TypedDict in function signature
```python
from typing import TypedDict, Unpack

class Options(TypedDict, total=False):
    timeout: int
    retries: int

def fetch(url: str, **kwargs: Unpack[Options]) -> str:
    return url

fetch("http://example.com", timeout=30)
```

### 11. Literal enum member
```python
from enum import Enum
from typing import Literal

class Status(Enum):
    OK = "ok"
    ERROR = "error"

def handle(s: Literal[Status.OK]) -> None:
    pass

handle(Status.OK)
```

### 12. Recursive type alias
```python
from typing import TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

def parse(data: JSON) -> JSON:
    return data
```

### 13. Contravariant TypeVar in Callable return
```python
from typing import TypeVar, Callable

T_contra = TypeVar("T_contra", contravariant=True)

def apply(f: Callable[[T_contra], None], val: T_contra) -> None:
    f(val)

def handler(x: object) -> None:
    pass

apply(handler, "string")
```

### 14. Overload with None
```python
from typing import overload

@overload
def process(x: None) -> None: ...
@overload
def process(x: int) -> str: ...

def process(x: int | None) -> str | None:
    if x is None:
        return None
    return str(x)

result = process(None)  # What's the type?
```

### 15. Final in inheritance
```python
from typing import Final

class Base:
    VALUE: Final[int] = 10

class Derived(Base):
    VALUE: int = 20  # Override Final - should error?
```

### 16. ParamSpec with *args only
```python
from typing import ParamSpec, Callable, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def partial_first(f: Callable[P, R], first: int) -> Callable[..., R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(first, *args, **kwargs)  # type: ignore
    return wrapper
```

### 17. Never in Union
```python
from typing import Never, NoReturn

def fails() -> Never:
    raise RuntimeError()

def maybe_fail(flag: bool) -> int:
    if flag:
        return fails()  # Never in return position
    return 42
```

### 18. Covariant container assignment
```python
from typing import Sequence

def process(items: Sequence[object]) -> None:
    pass

strings: list[str] = ["a", "b"]
process(strings)  # list[str] -> Sequence[object] covariance
```

### 19. Type narrowing with `in` operator
```python
def check(val: str | int, options: list[str]) -> str:
    if val in options:
        # Is val narrowed to str here?
        return val.upper()
    return str(val)
```

### 20. Generic bounds with Protocol
```python
from typing import TypeVar, Protocol

class Comparable(Protocol):
    def __lt__(self, other: object) -> bool: ...

T = TypeVar("T", bound=Comparable)

def minimum(a: T, b: T) -> T:
    return a if a < b else b

result = minimum(1, 2)  # int satisfies Comparable?
```
"""


# ============================================================================
# Stage 3: Code Tweaker - Modify code to trigger FP/FN
# ============================================================================

class CodeTweaker:
    """Tweaks original code from issues to potentially trigger false positives/negatives"""

    def __init__(self, github_token: str, model: str = "openai/gpt-4o"):
        self.github_token = github_token
        self.model = model
        self.agent = Agent(
            model=OpenAIChatModel(
                model,
                provider=GitHubProvider(api_key=github_token)
            ),
            system_prompt=f"""You are an expert at finding edge cases in Python type checkers (mypy, ty, pyrefly, zuban).

Your task is to TWEAK code from GitHub issues to cause DIVERGENCE between type checkers - 
where some report errors and others pass. This reveals false positives or false negatives.

TYPE CHECKERS WE TEST:
- mypy (python/mypy)
- ty (astral-sh/ty)  
- pyrefly (facebook/pyrefly)
- zuban (zubanls/zuban)

GOAL: Create code where AT LEAST ONE checker disagrees with the others.

IMPORTANT: 
- You will receive ORIGINAL CODE from a GitHub issue
- You must TWEAK that code, not generate completely new code
- Keep the structure similar, add/modify type annotations
- The code may have type bugs (that's fine) but syntax must be CORRECT

AREAS WITH KNOWN DIVERGENCE:
1. Protocol structural matching (especially with Callable)
2. Type narrowing in closures/nested functions
3. ClassVar in dataclasses
4. TypedDict with total=False and Required
5. Generic Self in classmethods
6. Callback signatures with default arguments
7. Type alias with forward references
8. Abstract property overridden as class attribute
9. Unpack with TypedDict (**kwargs typing)
10. Literal with Enum members
11. Recursive type aliases
12. Contravariant TypeVars
13. Overloads with None
14. Final override in subclass
15. ParamSpec edge cases
16. Never/NoReturn in unions
17. Covariant container assignment (list -> Sequence)
18. Type narrowing with `in` operator
19. Generic bounds with Protocol
20. TypeGuard/TypeIs narrowing scope

{DIVERGENCE_EXAMPLES}

RULES:
1. Code MUST have CORRECT SYNTAX (will be validated with compile())
2. Code MUST be SELF-CONTAINED with ALL imports
3. Use ONLY standard library (typing, abc, dataclasses, enum, collections.abc)
4. TWEAK the original code, don't replace it entirely
5. Type errors are OK, syntax errors are NOT

OUTPUT FORMAT:
```python
<complete code with correct syntax>
```
TWEAK_DESCRIPTION: <what you changed in the original code>""",
            deps_type=str,
        )

    async def tweak_code(self, issue: ExtractedIssue) -> TweakedExample:
        """Create initial tweak of the original code"""
        prompt = f"""TWEAK the following code from a GitHub issue to trigger type checker divergence.

ISSUE TRACKER: {issue.issue_tracker_url}
SPECIFIC ISSUE: {issue.issue_url}
TITLE: {issue.title}

ORIGINAL CODE TO TWEAK:
```python
{issue.original_code}
```

YOUR TASK:
1. START with the original code above
2. Make SMALL modifications to trigger divergence between type checkers (mypy, ty, pyrefly, zuban)
3. Keep the overall structure similar to the original
4. Add/modify type annotations using one of the known divergence patterns

DO NOT generate completely new code - MODIFY the existing code.

Possible modifications:
- Add type annotations where missing
- Change existing annotations to use Protocol, TypeGuard, Self, etc.
- Add a wrapper function with ParamSpec
- Modify return types to use overloads
- Add generic type parameters
- Wrap in a class with ClassVar or Final

Return the TWEAKED code in ```python ... ``` and TWEAK_DESCRIPTION: describing what you changed."""

        result = await self.agent.run(prompt, deps=self.github_token)
        response = str(result.output)
        code, description = self._parse_response(response)

        return TweakedExample(
            id=f"{issue.repo.replace('/', '-')}-{issue.issue_number}",
            source_issue_url=issue.issue_url,
            source_issue_tracker=issue.issue_tracker_url,
            original_code=issue.original_code,
            tweaked_code=code,
            tweak_description=description,
        )

    async def tweak_again(
        self, 
        example: TweakedExample, 
        validation: ValidationResult,
        round_num: int
    ) -> TweakedExample:
        """Tweak code again based on type checker results"""
        
        # Determine strategy based on current results
        all_error = all(c == "ERROR" for c in validation.checker_classifications.values())
        all_clean = all(c in ("CLEAN", "WARNING") for c in validation.checker_classifications.values())
        
        # Suggest different patterns based on round number
        patterns_to_try = [
            "Protocol structural matching with Callable",
            "Type narrowing in nested functions/closures",
            "ClassVar in dataclass",
            "TypedDict with Required and total=False",
            "Generic Self in classmethod",
            "Callback with default arguments",
            "Abstract property overridden as attribute",
            "Unpack with TypedDict for **kwargs",
            "Final override in subclass",
            "Covariant container (list[str] to Sequence[object])",
            "Type narrowing with `in` operator",
            "Overload with None return",
            "ParamSpec with partial application",
            "Never/NoReturn in return position",
            "Recursive type alias",
        ]
        suggested_pattern = patterns_to_try[(round_num - 1) % len(patterns_to_try)]
        
        if all_error:
            strategy = f"""ALL CHECKERS ERRORED - The code has actual type errors.

FIX the errors first, then apply pattern: {suggested_pattern}"""
        else:
            strategy = f"""ALL CHECKERS PASSED - Need to add a subtle edge case.

Apply pattern: {suggested_pattern}"""

        # Full error details
        error_details = "\n\n".join([
            f"=== {checker} ({cls}) ===\n{output}"
            for checker, (cls, output) in zip(
                validation.checker_classifications.keys(),
                zip(validation.checker_classifications.values(), validation.checker_outputs.values())
            )
        ])

        prompt = f"""Round {round_num}: No divergence yet. {strategy}

ORIGINAL CODE FROM ISSUE (for reference):
```python
{example.original_code}
```

CURRENT TWEAKED VERSION:
```python
{example.tweaked_code}
```

CURRENT RESULTS: {validation.checker_classifications}

ERRORS:
{error_details}

YOUR TASK:
1. Go back to the ORIGINAL CODE
2. Apply the pattern "{suggested_pattern}" to it
3. Make minimal changes - keep the structure of the original

Return the tweaked code in ```python ... ``` and TWEAK_DESCRIPTION: at the end."""

        result = await self.agent.run(prompt, deps=self.github_token)
        response = str(result.output)
        code, description = self._parse_response(response)

        return TweakedExample(
            id=example.id,
            source_issue_url=example.source_issue_url,
            source_issue_tracker=example.source_issue_tracker,
            original_code=example.original_code,
            tweaked_code=code,
            tweak_description=description,
        )

    async def fix_syntax(self, example: TweakedExample, error: str) -> TweakedExample:
        """Fix syntax errors in tweaked code"""
        prompt = f"""Fix the syntax error in this code:

ERROR: {error}

CODE:
```python
{example.tweaked_code}
```

Return the fixed code in ```python ... ```"""

        result = await self.agent.run(prompt, deps=self.github_token)
        response = str(result.output)
        code, _ = self._parse_response(response)

        return TweakedExample(
            id=example.id,
            source_issue_url=example.source_issue_url,
            source_issue_tracker=example.source_issue_tracker,
            original_code=example.original_code,
            tweaked_code=code,
            tweak_description=example.tweak_description,
        )

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse code and description from LLM response"""
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = response.strip()
            code = re.sub(r"^```\w*\s*", "", code)
            code = re.sub(r"\s*```$", "", code)

        desc_match = re.search(r"TWEAK_DESCRIPTION:\s*(.+)", response, re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else ""

        return code, description


# ============================================================================
# Stage 4: Type Checker Validator
# ============================================================================

class TypeCheckerValidator:
    """Runs type checkers and detects divergence"""

    CHECKERS = ["mypy", "pyrefly", "ty", "zuban"]

    def __init__(self):
        self._check_installed_checkers()

    def _check_installed_checkers(self):
        self.available_checkers = []
        for checker in self.CHECKERS:
            try:
                subprocess.run([checker, "--version"], capture_output=True, timeout=5)
                self.available_checkers.append(checker)
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        print(f"  Available type checkers: {', '.join(self.available_checkers)}")

    def validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Check if code is syntactically valid"""
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def run_type_checker(self, checker: str, code: str) -> str:
        """Run a type checker on the code"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Different checkers have different CLI syntax
            if checker == "ty":
                cmd = [checker, "check", temp_path]
            elif checker == "pyrefly":
                cmd = [checker, "check", temp_path]
            elif checker == "zuban":
                cmd = [checker, "check", temp_path]
            else:  # mypy
                cmd = [checker, temp_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except Exception as e:
            return f"ERROR: {e}"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def classify_output(self, output: str) -> str:
        """Classify type checker output as ERROR, WARNING, or CLEAN"""
        output_lower = output.lower()
        
        if "error" in output_lower:
            return "ERROR"
        elif "warning" in output_lower:
            return "WARNING"
        else:
            return "CLEAN"

    def check_divergence(self, classifications: Dict[str, str]) -> tuple[bool, str]:
        """Check if there's divergence between type checkers"""
        error_checkers = [c for c, cls in classifications.items() if cls == "ERROR"]
        clean_checkers = [c for c, cls in classifications.items() if cls in ("CLEAN", "WARNING")]

        # Divergence = some error AND some pass
        has_divergence = len(error_checkers) > 0 and len(clean_checkers) > 0

        if has_divergence:
            analysis = f"DIVERGENCE: {error_checkers} error, {clean_checkers} pass"
        elif len(error_checkers) == len(classifications):
            analysis = f"All checkers error: {error_checkers}"
        else:
            analysis = f"All checkers pass: {clean_checkers}"

        return has_divergence, analysis

    def validate(self, example: TweakedExample) -> ValidationResult:
        """Run all type checkers and check for divergence"""
        is_valid, syntax_error = self.validate_syntax(example.tweaked_code)
        
        if not is_valid:
            return ValidationResult(
                example_id=example.id,
                checker_outputs={},
                checker_classifications={},
                has_divergence=False,
                is_valid_syntax=False,
                analysis=f"Syntax error: {syntax_error}",
            )

        checker_outputs = {}
        classifications = {}

        for checker in self.available_checkers:
            output = self.run_type_checker(checker, example.tweaked_code)
            checker_outputs[checker] = output
            classifications[checker] = self.classify_output(output)

        has_divergence, analysis = self.check_divergence(classifications)

        return ValidationResult(
            example_id=example.id,
            checker_outputs=checker_outputs,
            checker_classifications=classifications,
            has_divergence=has_divergence,
            is_valid_syntax=True,
            analysis=analysis,
        )


# ============================================================================
# Main Pipeline
# ============================================================================

class FalseResultDiscoveryPipeline:
    """
    Pipeline to discover false positives/negatives in type checkers.
    
    Flow:
    1. Fetch closed issues with code from type checker repos
    2. For each issue, extract the original code
    3. Tweak the code (LLM) to try to trigger FP/FN
    4. Run type checkers
    5. If divergence -> success
    6. If no divergence -> tweak again, repeat
    """

    def __init__(
        self, 
        github_token: str, 
        model: str = "openai/gpt-4o",
        max_rounds: int = 10
    ):
        self.github_token = github_token
        self.model = model
        self.max_rounds = max_rounds

        self.issue_fetcher = IssueFetcher(github_token)
        self.code_tweaker = CodeTweaker(github_token, model)
        self.validator = TypeCheckerValidator()

    async def run(self, num_examples: int = 10) -> List[Dict]:
        """Run the pipeline"""
        print(f"Fetching {num_examples} issues from type checker repos...")

        issues = await self.issue_fetcher.fetch_issues_with_code(num_examples)
        
        if not issues:
            print("ERROR: No issues found")
            return []

        results = []

        for i, issue in enumerate(issues, 1):
            print(f"\n{'='*60}")
            print(f"[Issue {i}/{len(issues)}] {issue.issue_url}")
            
            # Check original code first
            original_example = TweakedExample(
                id=f"{issue.repo.replace('/', '-')}-{issue.issue_number}-original",
                source_issue_url=issue.issue_url,
                source_issue_tracker=issue.issue_tracker_url,
                original_code=issue.original_code,
                tweaked_code=issue.original_code,
                tweak_description="Original code from issue",
            )
            original_validation = self.validator.validate(original_example)
            
            if original_validation.has_divergence:
                print(f"  Original code has divergence!")
                print(f"\n  TYPE CHECKER REPORTS:")
                for checker, output in original_validation.checker_outputs.items():
                    classification = original_validation.checker_classifications[checker]
                    print(f"    {checker}: {classification}")
                    if output.strip():
                        lines = output.strip().split('\n')[:3]
                        for line in lines:
                            print(f"      {line}")
                
                print(f"\n  CODE:")
                print(f"  ```python")
                for line in issue.original_code.split('\n'):
                    print(f"  {line}")
                print(f"  ```")
                
                results.append({
                    "id": original_example.id,
                    "source_issue": issue.issue_url,
                    "source_issue_tracker": issue.issue_tracker_url,
                    "original_code": issue.original_code,
                    "tweaked_code": issue.original_code,
                    "tweak_description": "Original code from issue (no tweaking needed)",
                    "checker_outputs": original_validation.checker_outputs,
                    "checker_classifications": original_validation.checker_classifications,
                    "analysis": original_validation.analysis,
                    "rounds": 0,
                })
                continue

            # Stage 3: Initial tweak
            try:
                example = await self.code_tweaker.tweak_code(issue)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            # Stage 4-6: Validate and iterate until divergence
            divergence_found = False
            round_num = 0

            while round_num < self.max_rounds and not divergence_found:
                round_num += 1
                
                validation = self.validator.validate(example)

                # Handle syntax errors
                if not validation.is_valid_syntax:
                    print(f"  [Round {round_num}] Syntax error, fixing...")
                    example = await self.code_tweaker.fix_syntax(
                        example, validation.analysis
                    )
                    continue

                # Show round result
                print(f"\n  [Round {round_num}/{self.max_rounds}]")
                print(f"  Source: {example.source_issue_url}")
                
                # Show type checker reports
                print(f"\n  TYPE CHECKER REPORTS:")
                for checker, output in validation.checker_outputs.items():
                    classification = validation.checker_classifications[checker]
                    print(f"    {checker}: {classification}")
                    if output.strip():
                        # Show only first 3 lines of each checker's message
                        lines = output.strip().split('\n')[:3]
                        for line in lines:
                            print(f"      {line}")
                        if len(output.strip().split('\n')) > 3:
                            print(f"      ...")
                
                # Show tweaked code
                print(f"\n  TWEAKED CODE:")
                print(f"  ```python")
                for line in example.tweaked_code.split('\n'):
                    print(f"  {line}")
                print(f"  ```")

                if validation.has_divergence:
                    print(f"\n  ✓ DIVERGENCE: {validation.analysis}")
                    divergence_found = True
                else:
                    print(f"\n  No divergence - {validation.analysis}")
                    if round_num < self.max_rounds:
                        example = await self.code_tweaker.tweak_again(
                            example, validation, round_num
                        )

            # Save result
            if divergence_found:
                results.append({
                    "id": example.id,
                    "source_issue": example.source_issue_url,
                    "original_code": example.original_code,
                    "tweaked_code": example.tweaked_code,
                    "checker_outputs": validation.checker_outputs,
                    "checker_classifications": validation.checker_classifications,
                    "rounds": round_num,
                })

        # Summary
        print(f"\n{'='*60}")
        print(f"DONE: {len(results)} divergent examples found from {len(issues)} issues")
        print(f"{'='*60}")
        
        return results


# ============================================================================
# Output Functions
# ============================================================================

def export_to_json(results: List[Dict], output_path: str) -> None:
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


def print_results(results: List[Dict]) -> None:
    if not results:
        return

    print("\n" + "=" * 60)
    print("DIVERGENT EXAMPLES:")
    print("=" * 60)

    for r in results:
        print(f"\nSource: {r['source_issue']}")
        print(f"Classifications: {r['checker_classifications']}")
        print(f"\nTweaked code:")
        print("```python")
        print(r['tweaked_code'])
        print("```")
        print("-" * 60)


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Discover false positives/negatives in type checkers by tweaking code from closed issues"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="Model to use (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of issues to process (default: 10)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Max tweaking rounds per issue (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="divergent_examples.json",
        help="Output file (default: divergent_examples.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only save to JSON, don't print results",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        return

    if not validate_model(args.model):
        print(f"Warning: '{args.model}' not in known models, trying anyway...")

    token = os.environ.get("GITHUB_PAT")
    if not token:
        raise ValueError("Please set GITHUB_PAT environment variable")

    print(f"Using model: {args.model}")

    pipeline = FalseResultDiscoveryPipeline(
        github_token=token,
        model=args.model,
        max_rounds=args.max_rounds,
    )

    results = await pipeline.run(num_examples=args.num_examples)

    export_to_json(results, args.output)

    if not args.quiet:
        print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
