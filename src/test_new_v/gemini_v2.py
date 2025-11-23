from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
import re
import asyncio

@dataclass
class LLMState:
    """Shared state to track functions' results."""
    initial_prompt: bool = False
    urls_visited: bool = False
    issues_selected: bool = False
    code_tweaked: bool = False
    type_checkers_run: bool = False
    
    # Store results from each step
    selected_issues: list[str] = field(default_factory=list)
    code_examples: list[str] = field(default_factory=list)
    type_checker_outputs: dict = field(default_factory=dict)

def agent_prompt() -> str:
    """Create an initial prompt to the agent."""
    prompt = """Your job is to create code examples that force Python type checkers to generate false positive and false negative reports.
    
    A STRONG disagreement means: Some type checkers report "no issues found" while others report error messages.
    The more type checkers that disagree (different outputs), the stronger the example.
    
    Follow the functions in order and complete each step before moving to the next.
    After EACH tool call, you MUST call the corresponding evaluate function before proceeding."""
    return prompt

agent = Agent(
    'google-gla:gemini-2.0-flash',
    deps_type=LLMState,
    system_prompt=agent_prompt(),
)

@agent.tool
def visit_issue_trackers(ctx: RunContext[LLMState]) -> str:
    """
    Step 1: Visit the issue tracker URLs for mypy, pyrefly, zuban, and ty.
    Search for CLOSED issues from 2023-2025 with labels: 'bug', 'typechecking', or 'runtime semantics'.
    
    Use web_search to find at least 3-5 issues.
    
    Issue trackers:
    - mypy: https://github.com/python/mypy/issues
    - pyrefly: https://github.com/facebook/pyrefly/issues  
    - zuban: https://github.com/zubanls/zuban/issues
    - ty: https://github.com/astral-sh/ty/issues
    
    YOU MUST return the URLs in this EXACT format:
    
    FOUND ISSUES:
    - https://github.com/python/mypy/issues/XXXXX
    - https://github.com/python/mypy/issues/YYYYY
    - https://github.com/astral-sh/ty/issues/ZZZZZ
    
    After providing URLs, immediately call evaluate_visited_urls with your response.
    """
    if not ctx.deps.initial_prompt:
        return "ERROR: Initial prompt not acknowledged."
    
    return """Use web_search to find GitHub issues now. Search for:
    
    1. "github.com/python/mypy closed bug 2023"
    2. "github.com/astral-sh/ty closed bug 2023"  
    3. "github.com/facebook/pyrefly closed issues"
    4. "github.com/zubanls/zuban closed issues"
    
    Visit at least 3-5 issue URLs and return them in the FOUND ISSUES format.
    Then IMMEDIATELY call evaluate_visited_urls with your response."""

@agent.tool
def evaluate_visited_urls(ctx: RunContext[LLMState], urls_found: str) -> str:
    """
    Evaluator for visit_issue_trackers.
    Checks if the URLs are valid GitHub issue links from the correct repositories.
    
    Pass the exact output from visit_issue_trackers to this function.
    """
    # Check if we have valid URLs
    valid_repos = ['python/mypy', 'facebook/pyrefly', 'zubanls/zuban', 'astral-sh/ty']
    
    # Extract URLs from the response
    url_pattern = r'https://github\.com/([\w\-]+/[\w\-]+)/issues/(\d+)'
    found_urls = re.findall(url_pattern, urls_found)
    
    if len(found_urls) < 3:
        return f"EVALUATION FAILED: Need at least 3 valid issue URLs. Found only {len(found_urls)}. Search for more issues and call visit_issue_trackers again."
    
    # Check if URLs are from valid repositories
    valid_issues = []
    for repo, issue_num in found_urls:
        if any(valid_repo in repo for valid_repo in valid_repos):
            valid_issues.append(f"https://github.com/{repo}/issues/{issue_num}")
    
    if len(valid_issues) < 3:
        return f"EVALUATION FAILED: Need at least 3 URLs from valid repositories (mypy, pyrefly, zuban, ty). Found {len(valid_issues)}. Search again."
    
    # Evaluation passed
    ctx.deps.urls_visited = True
    ctx.deps.selected_issues = valid_issues
    return f"✓ EVALUATION PASSED: Found {len(valid_issues)} valid issue URLs.\nIssues: {', '.join(valid_issues)}\n\nProceed to NEXT STEP: Call select_and_extract_code"

@agent.tool
def select_and_extract_code(ctx: RunContext[LLMState]) -> str:
    """
    Step 2: From the selected issues, extract the code examples.
    Use web_fetch to visit each issue URL and extract the Python code examples.
    
    After extracting, call evaluate_extracted_code with your response.
    """
    if not ctx.deps.urls_visited:
        return "ERROR: Must visit URLs and pass evaluation first. Call visit_issue_trackers."
    
    return f"""Use web_fetch to visit these issue URLs and extract Python code:
    
    {chr(10).join(ctx.deps.selected_issues)}
    
    For EACH issue, find the Python code example and return in this format:
    
    ISSUE: [URL]
    CODE:
```python
    [code here]
```
    
    ---
    
    After extracting all code, IMMEDIATELY call evaluate_extracted_code with your response."""

@agent.tool
def evaluate_extracted_code(ctx: RunContext[LLMState], extracted_code: str) -> str:
    """
    Evaluator for select_and_extract_code.
    Checks if valid Python code was extracted.
    
    Pass the exact output from select_and_extract_code to this function.
    """
    # Check for code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', extracted_code, re.DOTALL)
    
    if len(code_blocks) < 3:
        return f"EVALUATION FAILED: Need at least 3 code examples. Found {len(code_blocks)}. Call select_and_extract_code again to extract more."
    
    # Check if code blocks are not empty and contain actual Python code
    non_empty = []
    for code in code_blocks:
        stripped = code.strip()
        # Basic check: should have at least one Python keyword or structure
        if stripped and any(keyword in stripped for keyword in ['def ', 'class ', 'import ', ':', '=', 'if ', 'for ', 'while ', 'return']):
            non_empty.append(stripped)
    
    if len(non_empty) < 3:
        return f"EVALUATION FAILED: Need at least 3 valid Python code blocks. Found {len(non_empty)}. Extract actual Python code."
    
    # Evaluation passed
    ctx.deps.issues_selected = True
    ctx.deps.code_examples = non_empty
    return f"✓ EVALUATION PASSED: Extracted {len(non_empty)} valid code examples.\n\nProceed to NEXT STEP: Call tweak_code_examples"

@agent.tool
def tweak_code_examples(ctx: RunContext[LLMState]) -> str:
    """
    Step 3: Tweak the code examples to create STRONG type checker disagreements.
    
    GOAL: Make some type checkers report "no issues" while others report errors.
    The more type checkers that disagree, the better!
    
    After tweaking, call evaluate_tweaked_code with your response.
    """
    if not ctx.deps.issues_selected:
        return "ERROR: Must extract code examples first. Call select_and_extract_code."
    
    return f"""You have {len(ctx.deps.code_examples)} code examples to tweak.
    
    For EACH example, modify it to create MAXIMUM disagreement between type checkers:
    - GOAL: Some checkers should find NO ERRORS (pass)
    - GOAL: Other checkers should find ERRORS (fail)
    - Exploit edge cases in type inference, generics, unions, protocols, etc.
    
    Format for EACH example:
    
    EXAMPLE 1:
    ORIGINAL ISSUE: [URL from {ctx.deps.selected_issues[0] if ctx.deps.selected_issues else 'N/A'}]
    TWEAKED CODE:
```python
    [modified code that will cause disagreements]
```
    TWEAKING STRATEGY: [Explain what you changed and WHY it causes disagreement]
    EXPECTED: Some checkers pass, some fail
    
    ---
    
    Do this for ALL {len(ctx.deps.code_examples)} examples.
    Then IMMEDIATELY call evaluate_tweaked_code with your response."""

@agent.tool
def evaluate_tweaked_code(ctx: RunContext[LLMState], tweaked_output: str) -> str:
    """
    Evaluator for tweak_code_examples.
    Checks if code was actually modified and explanations provided.
    
    Pass the exact output from tweak_code_examples to this function.
    """
    # Check for tweaked code blocks
    tweaked_blocks = re.findall(r'TWEAKED CODE:\s*```python\n(.*?)```', tweaked_output, re.DOTALL)
    
    if len(tweaked_blocks) < 3:
        return f"EVALUATION FAILED: Need at least 3 tweaked examples. Found {len(tweaked_blocks)}. Call tweak_code_examples again."
    
    # Check for tweaking strategy explanations
    strategy_count = len(re.findall(r'TWEAKING STRATEGY:', tweaked_output))
    
    if strategy_count < 3:
        return "EVALUATION FAILED: Must explain TWEAKING STRATEGY for each example. Call tweak_code_examples again."
    
    # Check for expected disagreement mentions
    expected_count = len(re.findall(r'EXPECTED:', tweaked_output))
    
    if expected_count < 3:
        return "EVALUATION FAILED: Must state EXPECTED disagreement for each example. Call tweak_code_examples again."
    
    # Evaluation passed
    ctx.deps.code_tweaked = True
    return f"✓ EVALUATION PASSED: {len(tweaked_blocks)} examples tweaked with strategies.\n\nProceed to NEXT STEP: Call run_type_checkers"

@agent.tool
def run_type_checkers(ctx: RunContext[LLMState]) -> str:
    """
    Step 4: Run mypy, pyrefly, ty, and zuban on EACH tweaked example.
    
    For each example:
    1. Save code to a .py file
    2. Run each type checker
    3. Capture outputs
    
    After running all checkers, call evaluate_type_checker_results with your response.
    """
    if not ctx.deps.code_tweaked:
        return "ERROR: Must tweak code first. Call tweak_code_examples."
    
    return """For EACH tweaked example, run ALL 4 type checkers.
    
    Use bash commands:
    1. echo 'code' > temp_example.py
    2. mypy temp_example.py
    3. pyrefly temp_example.py (or pyright if pyrefly not available)
    4. ty check temp_example.py
    5. zuban check temp_example.py (or pyre if zuban not available)
    
    Format for EACH example:
    
    EXAMPLE 1:
    CODE:
```python
    [tweaked code]
```
    
    MYPY OUTPUT:
    [exact output]
    
    PYREFLY OUTPUT:
    [exact output]
    
    TY OUTPUT:
    [exact output]
    
    ZUBAN OUTPUT:
    [exact output]
    
    DISAGREEMENT ANALYSIS: [How many passed vs failed]
    
    ---
    
    Then IMMEDIATELY call evaluate_type_checker_results with your response."""

@agent.tool
def evaluate_type_checker_results(ctx: RunContext[LLMState], results: str) -> str:
    """
    Evaluator for run_type_checkers.
    Checks if type checkers were run and there are REAL disagreements.
    
    DISAGREEMENT = Some checkers say "no issues" while others report errors.
    STRONG disagreement = 3 or 4 different outputs among the checkers.
    
    Pass the exact output from run_type_checkers to this function.
    """
    # Check for all type checker outputs
    checkers = ['MYPY OUTPUT:', 'PYREFLY OUTPUT:', 'TY OUTPUT:', 'ZUBAN OUTPUT:']
    
    for checker in checkers:
        count = results.count(checker)
        if count < 3:
            return f"EVALUATION FAILED: {checker} must appear at least 3 times (once per example). Found {count}. Call run_type_checkers again."
    
    # Parse examples and check for disagreements
    examples = re.split(r'EXAMPLE \d+:', results)[1:]
    
    if len(examples) < 3:
        return "EVALUATION FAILED: Need at least 3 examples with type checker outputs. Call run_type_checkers again."
    
    strong_disagreements = 0
    weak_disagreements = 0
    no_disagreements = 0
    
    for example in examples:
        outputs = {}
        for checker in ['MYPY', 'PYREFLY', 'TY', 'ZUBAN']:
            pattern = rf'{checker} OUTPUT:\s*(.+?)(?=(?:MYPY OUTPUT:|PYREFLY OUTPUT:|TY OUTPUT:|ZUBAN OUTPUT:|DISAGREEMENT ANALYSIS:|EXAMPLE|\Z))'
            match = re.search(pattern, example, re.DOTALL)
            if match:
                output = match.group(1).strip().lower()
                # Categorize as "pass" or "fail"
                if any(term in output for term in ['success', 'no issues', 'no error', 'passed', 'no problems']):
                    outputs[checker] = 'PASS'
                elif any(term in output for term in ['error', 'warning', 'issue', 'failed', 'found']):
                    outputs[checker] = 'FAIL'
                else:
                    outputs[checker] = output[:50]
        
        if len(outputs) < 4:
            continue
        
        # Count unique outputs
        unique_outputs = len(set(outputs.values()))
        
        # Check if there's a mix of PASS and FAIL
        has_pass = 'PASS' in outputs.values()
        has_fail = 'FAIL' in outputs.values()
        
        if has_pass and has_fail:
            if unique_outputs >= 3:
                strong_disagreements += 1
            else:
                weak_disagreements += 1
        else:
            no_disagreements += 1
    
    total_examples = len(examples)
    
    if strong_disagreements == 0 and weak_disagreements == 0:
        return f"EVALUATION FAILED: No disagreements found in {total_examples} examples. All type checkers agree. Call tweak_code_examples again with more aggressive modifications."
    
    if strong_disagreements < 2:
        return f"EVALUATION WEAK: Only {strong_disagreements} strong disagreements. Need at least 2. Call tweak_code_examples again to create stronger disagreements."
    
    # Evaluation passed
    ctx.deps.type_checkers_run = True
    ctx.deps.type_checker_outputs = {
        'strong': strong_disagreements,
        'weak': weak_disagreements,
        'none': no_disagreements,
        'results': results
    }
    
    return f"""✓ EVALUATION PASSED! 
    - Strong disagreements: {strong_disagreements} (mix of pass/fail with 3+ different outputs)
    - Weak disagreements: {weak_disagreements} (some pass, some fail)
    - No disagreements: {no_disagreements}
    
    Proceed to FINAL STEP: Call provide_final_output"""

@agent.tool
def provide_final_output(ctx: RunContext[LLMState]) -> str:
    """
    Step 5: Provide the final formatted output with all details.
    Include only examples with disagreements.
    """
    if not ctx.deps.type_checkers_run:
        return "ERROR: Must run type checkers first. Call run_type_checkers."
    
    stats = ctx.deps.type_checker_outputs
    
    return f"""Create final formatted output:
    
    ============================================================
    TYPE CHECKER DISAGREEMENT REPORT
    ============================================================
    
    Summary:
    - Strong Disagreements: {stats['strong']}
    - Weak Disagreements: {stats['weak']}
    - Total Examples Tested: {stats['strong'] + stats['weak'] + stats['none']}
    
    For EACH example with disagreement, show:
    
    EXAMPLE N:
    Original Issue: [GitHub URL]
    
    Code:
```python
    [tweaked code]
```
    
    Type Checker Results:
    ✓ MYPY: [output - PASS or FAIL with details]
    ✓ PYREFLY: [output - PASS or FAIL with details]
    ✓ TY: [output - PASS or FAIL with details]
    ✓ ZUBAN: [output - PASS or FAIL with details]
    
    Disagreement Strength: STRONG/WEAK
    Analysis: [Why this creates disagreement - which checkers disagree and why]
    
    ============================================================
    
    This is the FINAL output. Task complete!"""

async def main():
    """Main execution function."""
    state = LLMState()
    state.initial_prompt = True
    
    print("Starting Type Checker Disagreement Analysis...")
    print("=" * 80)
    
    # You can use these pre-found issues or let the agent search for new ones
    prefound_issues = """
    FOUND ISSUES:
    - https://github.com/python/mypy/issues/16531
    - https://github.com/python/mypy/issues/11497
    - https://github.com/astral-sh/ty/issues/938
    """
    
    result = await agent.run(
        f"""Complete ALL 9 steps of the type checker disagreement task:
        
        STEP 1: Call visit_issue_trackers (you can use these pre-found issues if helpful: {prefound_issues})
        STEP 2: Call evaluate_visited_urls
        STEP 3: Call select_and_extract_code
        STEP 4: Call evaluate_extracted_code
        STEP 5: Call tweak_code_examples
        STEP 6: Call evaluate_tweaked_code
        STEP 7: Call run_type_checkers
        STEP 8: Call evaluate_type_checker_results
        STEP 9: Call provide_final_output
        
        Execute EVERY step in order. After EACH function, call its corresponding evaluator.
        Do NOT skip any steps. Continue until you call provide_final_output.
        
        GOAL: Create code examples where some type checkers PASS (no errors) and others FAIL (report errors).
        The stronger the disagreement, the better!
        """,
        deps=state
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result.output)
    
    print("\n" + "=" * 80)
    print("EXECUTION STATE:")
    print("=" * 80)
    print(f"URLs Visited: {state.urls_visited}")
    print(f"Issues Selected: {state.issues_selected}")
    print(f"Code Tweaked: {state.code_tweaked}")
    print(f"Type Checkers Run: {state.type_checkers_run}")
    print(f"Selected Issues: {state.selected_issues}")
    
    if state.type_checker_outputs:
        print(f"\nDisagreement Stats:")
        print(f"  Strong: {state.type_checker_outputs.get('strong', 0)}")
        print(f"  Weak: {state.type_checker_outputs.get('weak', 0)}")
        print(f"  None: {state.type_checker_outputs.get('none', 0)}")

if __name__ == "__main__":
    asyncio.run(main())
