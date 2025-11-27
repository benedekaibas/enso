from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDenied
from dataclasses import dataclass, field
import asyncio

@dataclass
class LLMState:
    """Track workflow state."""
    found_issues: list[str] = field(default_factory=list)
    extracted_code: list[str] = field(default_factory=list)
    tweaked_code: list[str] = field(default_factory=list)
    attempt_count: int = 0
    checker_outputs: dict = field(default_factory=dict)

def system_prompt() -> str:
    """Initial prompt to the agent."""
    return """Your job is to create code examples that force Python type checkers to generate disagreements.

DISAGREEMENT: At least one type checker gives a different result from others.
Example: 3 checkers FAIL, 1 PASSES = DISAGREEMENT ✓

KEY STRATEGIES:
1. Generic type variance - Return wrong type when Generic[T] expected
2. Union narrowing - isinstance() then return mismatched type  
3. Async/await - Mix sync/async callables
4. Optional/None - Return None when non-Optional expected
5. Protocol typing - Partial protocol implementation

Example:
```python
from typing import TypeVar
T = TypeVar('T')
def foo(x: T) -> T:
    if isinstance(x, int):
        return "wrong"
    return x
```

Issue trackers:
- mypy: https://github.com/python/mypy/issues
- pyrefly: https://github.com/facebook/pyrefly/issues
- ty: https://github.com/astral-sh/ty/issues

Use bash_tool to run type checkers."""

agent = Agent(
    'google-gla:gemini-2.0-flash',
    deps_type=LLMState,
    system_prompt=system_prompt(),
)

@agent.tool
def find_issues(ctx: RunContext[LLMState], 
                issue_url_1: str, issue_url_2: str, issue_url_3: str) -> str:
    """Store 3 closed issue URLs from 2023-2025 with bug/typechecking labels."""
    print(f"\n{'='*60}")
    print(f"🔧 TOOL: find_issues")
    print(f"{'='*60}")
    print(f"  URL 1: {issue_url_1}")
    print(f"  URL 2: {issue_url_2}")
    print(f"  URL 3: {issue_url_3}")
    
    ctx.deps.found_issues = [issue_url_1, issue_url_2, issue_url_3]
    print("  ✓ Stored 3 issues")
    return "Stored 3 issues. Next: extract_code"

@agent.tool
def extract_code(ctx: RunContext[LLMState], 
                 code_1: str, code_2: str, code_3: str) -> str | ToolDenied:
    """Extract Python code from the 3 issues using web_fetch."""
    print(f"\n{'='*60}")
    print(f"🔧 TOOL: extract_code")
    print(f"{'='*60}")
    
    if not ctx.deps.found_issues:
        print("  ❌ ERROR: No issues found")
        return ToolDenied("Must find issues first. Call find_issues.")
    
    print(f"  Code 1: {len(code_1)} characters")
    print(f"  Code 2: {len(code_2)} characters")
    print(f"  Code 3: {len(code_3)} characters")
    
    ctx.deps.extracted_code = [code_1, code_2, code_3]
    print("  ✓ Extracted all 3 code examples")
    return "Extracted 3 code examples. Next: tweak_code"

@agent.tool
def tweak_code(ctx: RunContext[LLMState], 
               tweaked_1: str, tweaked_2: str, tweaked_3: str,
               strategy_1: str, strategy_2: str, strategy_3: str) -> str | ToolDenied:
    """Provide 3 tweaked versions using strategies from system prompt."""
    print(f"\n{'='*60}")
    print(f"🔧 TOOL: tweak_code (Attempt {ctx.deps.attempt_count + 1})")
    print(f"{'='*60}")
    
    if not ctx.deps.extracted_code:
        print("  ❌ ERROR: No extracted code")
        return ToolDenied("Must extract code first. Call extract_code.")
    
    ctx.deps.tweaked_code = [tweaked_1, tweaked_2, tweaked_3]
    ctx.deps.attempt_count += 1
    
    print(f"  Strategy 1: {strategy_1}")
    print(f"  Strategy 2: {strategy_2}")
    print(f"  Strategy 3: {strategy_3}")
    
    print(f"\n  Example 1 preview:")
    print(f"  {tweaked_1[:150]}...")
    print(f"\n  Example 2 preview:")
    print(f"  {tweaked_2[:150]}...")
    print(f"\n  Example 3 preview:")
    print(f"  {tweaked_3[:150]}...")
    
    print(f"\n  ✓ Created 3 tweaked examples")
    
    return f"""Tweaked 3 examples (attempt {ctx.deps.attempt_count}):
- Example 1: {strategy_1}
- Example 2: {strategy_2}
- Example 3: {strategy_3}

Next: run_type_checkers"""

@agent.tool
def run_type_checkers(ctx: RunContext[LLMState]) -> str | ToolDenied:
    """Run mypy, pyrefly, ty, and zuban on each example using bash_tool.
    
    FOR EACH EXAMPLE:
    1. bash_tool: cat > /tmp/test_N.py << 'EOF'
       [code]
       EOF
    2. bash_tool: mypy /tmp/test_N.py 2>&1
    3. bash_tool: pyrefly check /tmp/test_N.py 2>&1
    4. bash_tool: ty check /tmp/test_N.py 2>&1
    5. bash_tool: zuban check /tmp/test_N.py 2>&1
    
    Total: 15 bash_tool calls (3 files + 12 checker runs)
    Then call check_for_disagreements with all 12 outputs."""
    
    print(f"\n{'='*60}")
    print("🔧 TOOL: run_type_checkers")
    print(f"{'='*60}")
    
    if not ctx.deps.tweaked_code:
        print("  ❌ ERROR: No tweaked code")
        return ToolDenied("Must tweak code first. Call tweak_code.")
    
    print("  ⚠️  Agent must now use bash_tool 15 times:")
    print("      - 3 file creations (cat > /tmp/test_N.py)")
    print("      - 12 type checker runs (mypy, pyrefly, ty, zuban)")
    print("  ⚠️  Watch for bash_tool calls below...")
    
    return f"""Run type checkers on {len(ctx.deps.tweaked_code)} examples.
Use bash_tool to create files and run: mypy, pyrefly, ty, zuban
Then call check_for_disagreements."""

@agent.tool
def check_for_disagreements(ctx: RunContext[LLMState],
                            ex1_mypy: str, ex1_pyrefly: str, ex1_ty: str, ex1_zuban: str,
                            ex2_mypy: str, ex2_pyrefly: str, ex2_ty: str, ex2_zuban: str,
                            ex3_mypy: str, ex3_pyrefly: str, ex3_ty: str, ex3_zuban: str) -> str:
    """Check if at least one checker gives different results."""
    print(f"\n{'='*60}")
    print("🔧 TOOL: check_for_disagreements")
    print(f"{'='*60}")
    
    def categorize(output: str) -> str:
        """
        Categorize checker output as PASS or FAIL.
        
        PASS patterns (when no errors found):
        - mypy: "Success: no issues found in 1 source file"
        - pyrefly: "INFO 0 errors"
        - zuban: "Success: no issues found in 1 source file"
        - ty: "All checks passed!"
        
        Anything else = FAIL (errors found)
        """
        output_lower = output.lower() # lowering the output of the type checkers
        
        # Known PASS patterns
        pass_patterns = [
            'success: no issues found',  # mypy, zuban
            'info 0 errors',              # pyrefly
            'all checks passed',          # ty
        ]
        
        # If output matches any PASS pattern → PASS
        for pattern in pass_patterns:
            if pattern in output_lower:
                return 'PASS'
        
        # Anything else → FAIL
        return 'FAIL'
    
    examples = [
        (ex1_mypy, ex1_pyrefly, ex1_ty, ex1_zuban),
        (ex2_mypy, ex2_pyrefly, ex2_ty, ex2_zuban),
        (ex3_mypy, ex3_pyrefly, ex3_ty, ex3_zuban),
    ]
    
    ctx.deps.checker_outputs = {
        'ex1': examples[0],
        'ex2': examples[1],
        'ex3': examples[2],
    }
    
    disagreement_count = 0
    details = []
    
    print("\n  Analyzing results:")
    
    for i, (mypy, pyrefly, ty, zuban) in enumerate(examples, 1):
        results = {
            'mypy': categorize(mypy),
            'pyrefly': categorize(pyrefly),
            'ty': categorize(ty),
            'zuban': categorize(zuban),
        }
        
        unique = set(results.values())
        
        if len(unique) > 1:
            disagreement_count += 1
            print(f"  Example {i}: ✓ DISAGREEMENT")
            print(f"    mypy={results['mypy']}, pyrefly={results['pyrefly']}, ty={results['ty']}, zuban={results['zuban']}")
            details.append(f"Example {i}: DISAGREEMENT ✓ - {results}")
        else:
            print(f"  Example {i}: ✗ AGREEMENT")
            print(f"    All checkers: {list(results.values())[0]}")
            details.append(f"Example {i}: AGREEMENT ✗ - {results}")
    
    print(f"\n  {'='*60}")
    print(f"  Total disagreements: {disagreement_count}/3")
    print(f"  {'='*60}")
    
    if disagreement_count < 2:
        print("  ⚠️  INSUFFICIENT! Need at least 2 disagreements")
        print("  ⚠️  Will retry with different strategies...")
        return f"""INSUFFICIENT DISAGREEMENTS! Only {disagreement_count}/3.

{chr(10).join(details)}

Attempt {ctx.deps.attempt_count}/5. Try DIFFERENT strategies. Call tweak_code again."""
    
    print("  ✓ SUCCESS! Enough disagreements found")
    print("  ✓ Proceeding to final report...")
    return f"SUCCESS! Found {disagreement_count} disagreements.\n{chr(10).join(details)}\n\nNext: generate_final_report"

@agent.tool
def generate_final_report(ctx: RunContext[LLMState]) -> str | ToolDenied:
    """Generate final formatted report."""
    print(f"\n{'='*60}")
    print("🔧 TOOL: generate_final_report")
    print(f"{'='*60}")
    
    if not ctx.deps.checker_outputs:
        print("  ❌ ERROR: No checker outputs")
        return ToolDenied("Must run check_for_disagreements first.")
    
    print("  ✓ Generating final report...")
    
    def categorize(output: str) -> str:
        """Categorize output for display."""
        output_lower = output.lower()
        
        pass_patterns = [
            'success: no issues found',
            'info 0 errors',
            'all checks passed',
        ]
        
        for pattern in pass_patterns:
            if pattern in output_lower:
                return '✓ PASS'
        
        return '✗ FAIL'
    
    results = []
    results.append("\n" + "="*80)
    results.append("TYPE CHECKER DISAGREEMENT REPORT")
    results.append("="*80)
    
    disagreement_count = 0
    
    for i in range(1, 4):
        ex_key = f'ex{i}'
        mypy, pyrefly, ty, zuban = ctx.deps.checker_outputs[ex_key]
        
        results.append(f"\n{'='*80}")
        results.append(f"EXAMPLE {i}")
        results.append(f"{'='*80}")
        results.append(f"\nOriginal Issue: {ctx.deps.found_issues[i-1]}")
        results.append(f"\nCode:")
        results.append(f"```python")
        results.append(ctx.deps.tweaked_code[i-1])
        results.append(f"```")
        results.append(f"\nType Checker Results:")
        results.append(f"\n  MYPY: {categorize(mypy)}")
        results.append(f"    {mypy[:200]}")
        results.append(f"\n  PYREFLY: {categorize(pyrefly)}")
        results.append(f"    {pyrefly[:200]}")
        results.append(f"\n  TY: {categorize(ty)}")
        results.append(f"    {ty[:200]}")
        results.append(f"\n  ZUBAN: {categorize(zuban)}")
        results.append(f"    {zuban[:200]}")
        
        statuses = [categorize(mypy), categorize(pyrefly), categorize(ty), categorize(zuban)]
        unique = set(statuses)
        
        if len(unique) > 1:
            results.append(f"\n  ✓✓✓ DISAGREEMENT DETECTED ✓✓✓")
            disagreement_count += 1
        else:
            results.append(f"\n  ✗ NO DISAGREEMENT")
    
    results.append("\n" + "="*80)
    results.append(f"Summary: {disagreement_count}/3 examples show disagreements")
    results.append(f"Attempts: {ctx.deps.attempt_count}")
    results.append("="*80)
    
    print(f"  ✓ Report generated successfully!")
    
    return "\n".join(results)

async def main():
    state = LLMState()
    
    print("\n" + "="*80)
    print("STARTING TYPE CHECKER DISAGREEMENT ANALYSIS")
    print("="*80 + "\n")
    
    result = await agent.run(
        """Complete these steps:

1. Find 3 closed bug issues from 2023-2025
2. Call find_issues with 3 URLs
3. Extract code with web_fetch
4. Call extract_code with 3 examples
5. Tweak code to cause disagreements
6. Call tweak_code with 3 examples + 3 strategies
7. Run bash_tool on EACH example (15 total calls)
8. Call check_for_disagreements with 12 outputs
9. If < 2 disagreements: go back to step 5
10. Call generate_final_report
11. Return the full report

CRITICAL: In step 7, actually use bash_tool for each example.""",
        deps=state
    )
    
    print("\n\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"\nFinal Attempts: {state.attempt_count}")
    print(f"Issues Found: {len(state.found_issues)}")
    print(f"Code Examples: {len(state.tweaked_code)}")
    
    if state.found_issues:
        print("\nIssue URLs:")
        for i, url in enumerate(state.found_issues, 1):
            print(f"  {i}. {url}")
    
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
