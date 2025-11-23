from pydantic_ai import Agent, RunContext
import asyncio
import re


def system_prompt() -> str:
    """Prompt that we pass to the LLM."""
    prompt = """
    Your job is to create code examples that force Python type checkers to generate false positive and false negative reports.
    For that you have to do the following tasks:
    1. Visit the URLs provided to the issue trackers of the type checkers.
    2. Randomly select closed issues from 2023-2025 randomly (only select issues where the label is bug or typechecking or runtime semantics).
    3. Take the code examples from the issue trackers and tweak those examples in a way that would force the newest version of type checkers
    (mypy, pyrefly, ty, and zuban) to report false positive or false negative.
    4. After creating these examples run the type checkers (mypy, pyrefly, ty, and zuban) with the versions on my computer on them and see if 
    there is a disagreement between the type checkers. If there is a disagreement then output the code example in the terminal, if there is no 
    disagreement then tweak the code example until there is a disagreement between the report of the type checkers.
    5. Give the code examples in the output and also show the link where the code examples were taken from for each generated code and also show
    the output of the different type checkers when you run them on the code examples.
    """
    return prompt

agent = Agent(
    'google-gla:gemini-2.0-flash',  
    system_prompt=system_prompt(),
)


@agent.tool
def number_of_code_examples() -> int:
    """The number of code examples we want to generate."""
    return 3

@agent.tool
def issue_trackers() -> dict:
    """Link to the issue trackers."""
    return {
        "mypy": "https://github.com/python/mypy/issues",
        "pyrefly": "https://github.com/facebook/pyrefly/issues",
        "zuban": "https://github.com/zubanls/zuban/issues",
        "ty": "https://github.com/astral-sh/ty/issues",
    }

@agent.tool_plain
def format_output(raw_output: str) -> str:
    """Formatting the output of the LLM."""
    lines = []
    lines.append("=" * 80)
    lines.append("TYPE CHECKER DISAGREEMENT EXAMPLES")
    lines.append("=" * 80)
    lines.append("")
    
    examples = re.split(r'EXAMPLE \d+', raw_output)
    
    for i, example in enumerate(examples[1:], 1):  # Skip first empty split
        lines.append(f"{'=' * 80}")
        lines.append(f"EXAMPLE #{i}")
        lines.append(f"{'=' * 80}")
        lines.append("")
        
        link_match = re.search(r'LINK:\s*(.+)', example)
        if link_match:
            lines.append("📎 Original Issue:")
            lines.append(f"   {link_match.group(1).strip()}")
            lines.append("")
        
        code_match = re.search(r'CODE:\s*```python\n(.*?)```', example, re.DOTALL)
        if code_match:
            lines.append("💻 Code:")
            lines.append("-" * 80)
            lines.append(code_match.group(1).strip())
            lines.append("-" * 80)
            lines.append("")
        
        lines.append("🔍 Type Checker Reports:")
        lines.append("")
        
        for checker in ['MYPY', 'PYREFLY', 'TY', 'ZUBAN']:
            pattern = rf'{checker}:\s*(.+?)(?=(?:MYPY:|PYREFLY:|TY:|ZUBAN:|EXAMPLE|\Z))'
            checker_match = re.search(pattern, example, re.DOTALL)
            if checker_match:
                lines.append(f"  [{checker}]")
                lines.append(f"  {'-' * 76}")
                output = checker_match.group(1).strip()
                for line in output.split('\n'):
                    lines.append(f"  {line}")
                lines.append("")
        
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

@agent.tool
async def main():
    """Run the agent."""
    # Prepare the user message with context
    trackers = issue_trackers()
    num_examples = number_of_code_examples()
    
    user_message = f"""
    Please generate {num_examples} code examples that create disagreements between type checkers.
    
    Issue tracker URLs:
    {trackers}
    
    Remember to:
    - Find closed bug issues from 2023-2025
    - Create or tweak examples to cause disagreements
    - Run all type checkers and show their outputs
    - Provide links to original issues
    - The format of the output should be:

        EXAMPLE N
        LINK: [url to original issue]
        CODE:
        ```python
        [the code]
        ```
        [ Output of the type checkers:
        MYPY: [output]
        PYREFLY: [output]
        TY: [output]
        ZUBAN: [output]
        ]
    """
 
    result = agent.run_sync(user_message)
    
    print(result.output)
    
    return result

if __name__ == "__main__":
    # Run the async main function
    result = asyncio.run(main())

