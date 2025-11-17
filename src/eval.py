import json
import os
import glob
import subprocess
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel


class TypeCheckerResult(BaseModel):
    """Result from a single type checker."""
    checker_name: str
    status: str  # "PASS", "FAIL", or "TOOL_MISSING"
    return_code: Optional[int]
    output: str


class CodeAnalysis(BaseModel):
    """Analysis of the expected type checking behavior."""
    correct_behavior: str = Field(
        description="Expected behavior: 'PASS' if code should type check, 'FAIL' if it should not"
    )
    reasoning: str = Field(
        description="Detailed explanation of why this is the correct behavior"
    )
    type_system_feature: str = Field(
        description="The Python typing feature being tested (e.g., 'NewType variance', 'Protocol defaults')"
    )
    correct_checkers: List[str] = Field(
        description="List of type checkers that behaved correctly"
    )
    incorrect_checkers: List[str] = Field(
        description="List of type checkers that behaved incorrectly"
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'"
    )


class FileEvaluation(BaseModel):
    """Complete evaluation for a single Python file."""
    filename: str
    code: str
    checker_results: List[TypeCheckerResult]
    analysis: CodeAnalysis


def find_latest_snippets_dir() -> Optional[str]:
    """Find the most recent extracted snippets directory."""
    candidates = glob.glob("output_*/extracted_python_snippets")
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def run_type_checkers(files_to_check: List[str]) -> Dict[str, List[TypeCheckerResult]]:
    """Run all type checkers on the given files."""
    type_checkers = [
        ("mypy", ["mypy"]),
        ("pyrefly", ["pyrefly", "check"]),
        ("zuban", ["zuban", "check"]),
        ("ty", ["ty", "check"]),
    ]

    all_results: Dict[str, List[TypeCheckerResult]] = {}
    print("\nRunning type checkers")

    for filename in sorted(files_to_check):
        print(f"\nProcessing {os.path.basename(filename)}")
        file_results: List[TypeCheckerResult] = []
        
        for name, base_command in type_checkers:
            cmd = base_command + [filename]
            print(f"  -> {name}")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                output = (result.stdout or "") + (result.stderr or "")
                status = "PASS" if result.returncode == 0 else "FAIL"
                
                file_results.append(TypeCheckerResult(
                    checker_name=name,
                    status=status,
                    return_code=result.returncode,
                    output=output
                ))
                
                print(f"     Status: {status}")
                if status == "FAIL" and output.strip():
                    print(f"     Error: {output.strip()[:100]}...")
                    
            except FileNotFoundError:
                file_results.append(TypeCheckerResult(
                    checker_name=name,
                    status="TOOL_MISSING",
                    return_code=None,
                    output=f"{name} not found"
                ))
                print(f"     {name} not found")
                
        all_results[filename] = file_results
        
    return all_results


def create_analysis_agent() -> Agent[None, CodeAnalysis]:
    """Create Pydantic AI agent for type checking analysis using GitHub Models."""
    agent = Agent(
        'openai:gpt-4o',
        output_type=CodeAnalysis,
        system_prompt="""
You are an expert in Python's type system and type checkers (mypy, pyright, etc.).
Your task is to analyze Python code snippets and determine:
1. Whether the code SHOULD pass type checking (PASS) or fail (FAIL) according to Python typing standards
2. Which type checkers are behaving correctly based on their results
3. What typing feature is being tested
Consider these Python typing specifications:
- PEP 484 (Type Hints), 544 (Protocols), 586 (Literal Types)
- PEP 589 (TypedDict), 591 (Final), 612 (Parameter), 613 (TypeAlias)
- PEP 655 (Required/NotRequired), 673 (Self), 675 (Arbitrary Literal String)
- PEP 692 (TypedDict **kwargs)
- Variance rules (covariant, contravariant, invariant)
- Protocol structural subtyping semantics
- NewType nominal typing semantics
- TypeGuard and type narrowing behavior
- Final and override semantics
- Overload resolution rules
Be precise and explain your reasoning clearly.
"""
    )
    return agent


async def analyze_with_agent(    agent: Agent[None, CodeAnalysis],
    code: str,
    checker_results: List[TypeCheckerResult]
) -> CodeAnalysis:
    """Use Pydantic AI agent to analyze the code and determine correct behavior."""
    
    results_summary = "\n".join([
        f"- {r.checker_name}: {r.status}"
        for r in checker_results
        if r.status != "TOOL_MISSING"
    ])
    
    detailed_outputs = "\n\n".join([
        f"### {r.checker_name}\nStatus: {r.status}\nOutput:\n{r.output[:500]}"
        for r in checker_results
        if r.status != "TOOL_MISSING" and r.output.strip()
    ])
    
    prompt = f"""Analyze this Python code snippet:

## Code:
```python
{code}
```

## Type Checker Results:
{results_summary}

## Detailed Error Messages:
{detailed_outputs}

Determine:
1. The correct expected behavior (PASS or FAIL)
2. Which checkers are correct/incorrect
3. What typing feature is being tested
4. Your confidence level"""

    result = await agent.run(prompt)
    return result.output


async def main() -> int:
    """Main evaluation pipeline using Pydantic AI with GitHub Models."""
    
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")
    if not github_token:
        print("Error: GITHUB_TOKEN or GITHUB_PAT environment variable not set")
        print("Get your token from: https://github.com/settings/tokens")
        return 1
    
    os.environ["OPENAI_API_KEY"] = github_token
    os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com"
    
    agent = create_analysis_agent()
    
    snippets_dir = find_latest_snippets_dir()
    if not snippets_dir:
        print("No extracted_python_snippets directory found under output_*")
        return 1

    py_files = sorted(glob.glob(os.path.join(snippets_dir, "*.py")))
    if not py_files:
        print(f"No .py files found in {snippets_dir}")
        return 1

    print(f"Using snippets directory: {snippets_dir}")
    
    checker_results = run_type_checkers(py_files)
    
    print("\n--- PYDANTIC AI EVALUATION ---")
    evaluations: List[FileEvaluation] = []
    
    for filename in py_files:
        print(f"\nAnalyzing {os.path.basename(filename)}...")
        
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
        
        results = checker_results[filename]
        analysis = await analyze_with_agent(agent, code, results)
        
        evaluation = FileEvaluation(
            filename=filename,
            code=code,
            checker_results=results,
            analysis=analysis
        )
        evaluations.append(evaluation)
        
        print(f"  Expected: {analysis.correct_behavior}")
        print(f"  Feature: {analysis.type_system_feature}")
        print(f"  Correct checkers: {', '.join(analysis.correct_checkers)}")
        print(f"  Incorrect checkers: {', '.join(analysis.incorrect_checkers)}")
        print(f"  Confidence: {analysis.confidence}")
    
    parent = os.path.dirname(snippets_dir)
    eval_file = os.path.join(parent, "pydantic_ai_evaluations.json")
    
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(
            [eval.model_dump() for eval in evaluations],
            f,
            indent=2
        )
    print(f"\nSaved Pydantic AI evaluations to {eval_file}")
    
    print("\n--- SUMMARY ---")
    print(f"Total files analyzed: {len(evaluations)}")
    
    high_confidence = sum(1 for e in evaluations if e.analysis.confidence == "high")
    print(f"High confidence evaluations: {high_confidence}")
    
    checker_correct_counts: Dict[str, int] = {}
    checker_total_counts: Dict[str, int] = {}
    
    for eval in evaluations:
        for checker in eval.analysis.correct_checkers:
            checker_correct_counts[checker] = checker_correct_counts.get(checker, 0) + 1
            checker_total_counts[checker] = checker_total_counts.get(checker, 0) + 1
        for checker in eval.analysis.incorrect_checkers:
            checker_total_counts[checker] = checker_total_counts.get(checker, 0) + 1
    
    print("\nType Checker Accuracy:")
    for checker in sorted(checker_total_counts.keys()):
        correct = checker_correct_counts.get(checker, 0)
        total = checker_total_counts[checker]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"  {checker}: {correct}/{total} ({accuracy:.1f}%)")
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit(asyncio.run(main()))
