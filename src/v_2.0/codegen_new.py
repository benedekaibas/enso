from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, HttpUrl
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider
from dataclasses import dataclass
import os
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import re
import httpx
from bs4 import BeautifulSoup

# ============================================================================
# Data Models
# ============================================================================

class TypeCheckerIssue(BaseModel):
    """Represents a known type checker divergence issue"""
    title: str
    issue_url: str
    checker_behaviors: Dict[str, str]  # checker -> behavior description
    category: str  # e.g., "Protocol", "TypeGuard", "Generics"
    description: str

class CodeExample(BaseModel):
    """A generated code example demonstrating type checker divergence"""
    id: str
    category: str
    code: str
    expected_behaviors: Dict[str, str]  # checker -> expected behavior
    reason: str
    issue_references: List[str] = []

class ValidationResult(BaseModel):
    """Result of running type checkers on an example"""
    example_id: str
    checker_outputs: Dict[str, str]  # checker -> output
    has_divergence: bool
    is_valid_syntax: bool
    error_message: Optional[str] = None

# ============================================================================
# Web Search Helper (GitHub Models Compatible)
# ============================================================================

class WebSearchHelper:
    """Manual web search helper for GitHub Models compatibility"""
    
    @staticmethod
    async def search_github_issues(query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search GitHub issues using GitHub's search API"""
        results = []
        
        # Use GitHub's search API
        search_url = "https://api.github.com/search/issues"
        params = {
            "q": f"{query} type:issue",
            "sort": "relevance",
            "per_page": max_results,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(search_url, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                for item in data.get("items", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("html_url", ""),
                        "body": item.get("body", "")[:500],  # First 500 chars
                    })
        except Exception as e:
            print(f"      Search error: {e}")
        
        return results
    
    @staticmethod
    async def fetch_url_content(url: str) -> Optional[str]:
        """Fetch and extract text content from a URL"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0, follow_redirects=True)
                response.raise_for_status()
                
                # Parse HTML and extract text
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # Limit to first 5000 chars
        except Exception as e:
            print(f"      Error fetching {url}: {e}")
            return None

# ============================================================================
# Stage 1: Issue Research Agent
# ============================================================================

class IssueResearchAgent:
    """Searches type checker issue trackers for known divergences"""
    
    SEARCH_QUERIES = [
        "protocol default arguments mypy type checker",
        "TypeGuard narrowing mypy pyright",
        "TypedDict total inheritance type checker",
        "ParamSpec decorator mypy",
        "Self type Generic bound",
        "NewType variance type checker",
        "overload literal mypy",
        "Final property override",
        "TypeVar bound inheritance",
        "Protocol keyword arguments type checker",
        "typing variance contravariant",
        "ClassVar type checker",
        "Callable ParamSpec",
        "Generic type parameter",
        "Union narrowing isinstance",
    ]
    
    def __init__(self, github_token: str, model: str = "openai/gpt-4o"):
        """Initialize with GitHub token for API access"""
        self.github_token = github_token
        self.model = model
        self.web_helper = WebSearchHelper()
        
        # Initialize agent with GitHubProvider (correct syntax)
        self.agent = Agent(
            model=OpenAIChatModel(
                model,
                provider=GitHubProvider(api_key=github_token)
            ),
            system_prompt="""You are an expert at analyzing Python type checker issues.
            You will be provided with GitHub issue search results and content.
            
            For each divergence you find, extract:
            1. A clear title describing the issue
            2. The specific type feature causing divergence
            3. How EACH checker (mypy, pyrefly, ty, zuban) behaves differently
            4. The GitHub issue URL
            5. The category (e.g., Protocol, TypeGuard, Generics, etc.)
            6. Technical explanation of why they diverge
            
            Return results as a JSON array of objects with this structure:
            {
              "title": "Brief description",
              "issue_url": "https://github.com/...",
              "checker_behaviors": {
                "mypy": "accepts/rejects with reason",
                "pyrefly": "inferred behavior based on similarity to pyright",
                "ty": "inferred behavior",
                "zuban": "inferred behavior based on similarity to pyre"
              },
              "category": "Protocol|TypeGuard|Generics|etc",
              "description": "Technical explanation"
            }
            
            Since pyrefly, ty, and zuban may not be mentioned in issues, infer their likely
            behavior based on the type checker they're similar to. Focus on issues showing
            clear disagreement between type checkers.""",
            deps_type=str,  # GitHub token
        )
    
    async def search_issues(self, max_issues: int = 10) -> List[TypeCheckerIssue]:
        """Search issue trackers for type checker divergences"""
        all_issues = []
        
        print(f"  Searching with {len(self.SEARCH_QUERIES)} queries...")
        
        for idx, query in enumerate(self.SEARCH_QUERIES[:max_issues * 2], 1):
            print(f"    Query {idx}: {query[:50]}...")
            
            try:
                # Step 1: Search GitHub issues
                search_results = await self.web_helper.search_github_issues(query, max_results=3)
                
                if not search_results:
                    print("      No results found")
                    continue
                
                # Step 2: Fetch full content for top results
                enriched_results = []
                for result in search_results[:2]:  # Top 2 results
                    full_content = await self.web_helper.fetch_url_content(result["url"])
                    if full_content:
                        result["full_content"] = full_content
                        enriched_results.append(result)
                
                if not enriched_results:
                    print("      Could not fetch content")
                    continue
                
                # Step 3: Ask LLM to extract structured information
                prompt = f"""Analyze these GitHub issue search results for type checker divergences:

Query: "{query}"

Search Results:
{json.dumps(enriched_results, indent=2)[:3000]}

Extract 1-2 type checker divergence issues from this content. Return as a JSON array.
Focus on actual behavioral differences between type checkers."""
                
                result = await self.agent.run(
                    prompt,
                    deps=self.github_token,
                )
                
                # Parse the LLM response
                response_text = str(result.output)
                issues = self._parse_issues_from_response(response_text)
                all_issues.extend(issues)
                
                print(f"      Found {len(issues)} issues from this query")
                
                # Stop if we have enough
                if len(all_issues) >= max_issues:
                    break
                    
            except Exception as e:
                print(f"      Error processing query: {e}")
                continue
        
        # Deduplicate by URL
        unique_issues = {issue.issue_url: issue for issue in all_issues}
        final_issues = list(unique_issues.values())[:max_issues]
        
        print(f"  Total unique issues found: {len(final_issues)}")
        return final_issues
    
    def _parse_issues_from_response(self, response_text: str) -> List[TypeCheckerIssue]:
        """Parse TypeCheckerIssue objects from LLM response"""
        issues = []
        
        # Try to find JSON in the response
        json_pattern = r'\[[\s\S]*?\]|\{[\s\S]*?\}'
        matches = re.finditer(json_pattern, response_text)
        
        for match in matches:
            try:
                json_text = match.group(0)
                data = json.loads(json_text)
                
                # Handle both single object and array
                if isinstance(data, dict):
                    data = [data]
                
                for item in data:
                    if self._is_valid_issue_dict(item):
                        issue = TypeCheckerIssue(**item)
                        issues.append(issue)
            except (json.JSONDecodeError, Exception):
                continue
        
        return issues
    
    def _is_valid_issue_dict(self, item: dict) -> bool:
        """Check if a dictionary has required fields for TypeCheckerIssue"""
        required = {"title", "issue_url", "checker_behaviors", "category", "description"}
        return all(key in item for key in required)

# ============================================================================
# Stage 2: Code Generation Agent
# ============================================================================

class CodeGenerationAgent:
    """Generates code examples based on researched issues"""
    
    def __init__(self, github_token: str, model: str = "openai/gpt-4o"):
        self.github_token = github_token
        self.agent = Agent(
            model=OpenAIChatModel(
                model,
                provider=GitHubProvider(api_key=github_token)
            ),
            system_prompt="""You are an expert Python developer specializing in type systems.
            Generate COMPLETE, RUNNABLE Python code examples that demonstrate type checker
            divergences based on real issues.
            
            CRITICAL REQUIREMENTS:
            1. Code must be syntactically valid Python 3.11+
            2. Include ALL necessary imports
            3. Must be self-contained and runnable
            4. Use only standard library + typing/typing_extensions
            5. Include if __name__ == "__main__": block
            6. No forward reference issues
            7. Clear comments explaining the divergence
            8. Focus on divergences between mypy, pyrefly, ty, and zuban
            
            OUTPUT FORMAT:
            # id: <category>-<specific-case>
            # EXPECTED:
            #   mypy: <specific behavior or error>
            #   pyrefly: <specific behavior or error>
            #   ty: <specific behavior or error>
            #   zuban: <specific behavior or error>
            # REASON: <technical explanation>
            
            <complete runnable code>
            """,
            deps_type=str,
        )
    
    async def generate_example(self, issue: TypeCheckerIssue) -> CodeExample:
        """Generate a code example for a specific issue"""
        prompt = f"""Generate a Python code example demonstrating this type checker divergence:

Title: {issue.title}
Category: {issue.category}
Description: {issue.description}

Known behaviors:
{json.dumps(issue.checker_behaviors, indent=2)}

Reference: {issue.issue_url}

Create a MINIMAL, COMPLETE, RUNNABLE example that clearly shows this divergence.
The code must compile and run without errors."""

        result = await self.agent.run(prompt, deps=self.github_token)
        
        # Parse the result into a CodeExample
        code_text = result.output
        
        return CodeExample(
            id=f"{issue.category}-example",
            category=issue.category,
            code=code_text,
            expected_behaviors=issue.checker_behaviors,
            reason=issue.description,
            issue_references=[issue.issue_url],
        )

# ============================================================================
# Stage 3: Validation Agent
# ============================================================================

class ValidationAgent:
    """Validates generated code by running actual type checkers"""
    
    CHECKERS = ["pyrefly", "ty", "zuban", "mypy"]
    
    def __init__(self):
        """Initialize validation agent"""
        self._check_installed_checkers()
    
    def _check_installed_checkers(self):
        """Check which type checkers are available"""
        self.available_checkers = []
        for checker in self.CHECKERS:
            try:
                subprocess.run(
                    [checker, "--version"],
                    capture_output=True,
                    timeout=5,
                )
                self.available_checkers.append(checker)
            except (subprocess.SubprocessError, FileNotFoundError):
                print(f"Warning: {checker} not installed")
    
    def validate_syntax(self, code: str) -> tuple[bool, Optional[str]]:
        """Check if code is syntactically valid"""
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def run_type_checker(self, checker: str, code: str) -> str:
        """Run a specific type checker on the code"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [checker, temp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return f"Timeout running {checker}"
        except Exception as e:
            return f"Error running {checker}: {e}"
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def validate_example(self, example: CodeExample) -> ValidationResult:
        """Validate a code example by running all type checkers"""
        # Check syntax first
        is_valid, syntax_error = self.validate_syntax(example.code)
        if not is_valid:
            return ValidationResult(
                example_id=example.id,
                checker_outputs={},
                has_divergence=False,
                is_valid_syntax=False,
                error_message=f"Syntax error: {syntax_error}",
            )
        
        # Run type checkers
        checker_outputs = {}
        for checker in self.available_checkers:
            output = self.run_type_checker(checker, example.code)
            checker_outputs[checker] = output
        
        # Check if there's actual divergence
        unique_outputs = set(checker_outputs.values())
        has_divergence = len(unique_outputs) > 1
        
        return ValidationResult(
            example_id=example.id,
            checker_outputs=checker_outputs,
            has_divergence=has_divergence,
            is_valid_syntax=True,
            error_message=None,
        )

# ============================================================================
# Stage 4: Refinement Agent
# ============================================================================

class RefinementAgent:
    """Refines examples that don't show divergence or have issues"""
    
    def __init__(self, github_token: str, model: str = "openai/gpt-4o"):
        self.github_token = github_token
        self.agent = Agent(
            model=OpenAIChatModel(
                model,
                provider=GitHubProvider(api_key=github_token)
            ),
            system_prompt="""You are an expert at refining Python type checker examples.
            Given a code example and validation results, improve it to:
            1. Fix any syntax errors
            2. Make type checker divergence more apparent
            3. Ensure it's still minimal and focused
            4. Keep it runnable and self-contained
            5. Target divergences between mypy, pyrefly, ty, and zuban
            
            If the example doesn't show divergence, adjust the code to make the
            type issue more prominent while staying true to the original issue.""",
            deps_type=str,
        )
    
    async def refine_example(
        self,
        example: CodeExample,
        validation: ValidationResult,
    ) -> CodeExample:
        """Refine an example based on validation results"""
        prompt = f"""Refine this code example:

Original Example ID: {example.id}
Category: {example.category}

CODE:
```python
{example.code}
```

VALIDATION RESULTS:
- Valid syntax: {validation.is_valid_syntax}
- Has divergence: {validation.has_divergence}
{f"- Error: {validation.error_message}" if validation.error_message else ""}

Type checker outputs:
{json.dumps(validation.checker_outputs, indent=2)}

Expected behaviors:
{json.dumps(example.expected_behaviors, indent=2)}

Reason: {example.reason}

{"Fix the syntax errors and ensure the code runs." if not validation.is_valid_syntax else ""}
{"Make the type checker divergence more apparent." if not validation.has_divergence else ""}

Return the COMPLETE refined code with all comments and structure."""

        result = await self.agent.run(prompt, deps=self.github_token)
        
        return CodeExample(
            id=f"{example.id}-refined",
            category=example.category,
            code=result.output,
            expected_behaviors=example.expected_behaviors,
            reason=example.reason,
            issue_references=example.issue_references,
        )

# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

class TypeCheckerDivergencePipeline:
    """Orchestrates the entire pipeline"""
    
    def __init__(
        self,
        github_token: str,
        model: str = "openai/gpt-4o",
        max_refinement_rounds: int = 3,
    ):
        self.github_token = github_token
        self.model = model
        self.max_refinement_rounds = max_refinement_rounds
        
        self.research_agent = IssueResearchAgent(github_token, model)
        self.generation_agent = CodeGenerationAgent(github_token, model)
        self.validation_agent = ValidationAgent()
        self.refinement_agent = RefinementAgent(github_token, model)
    
    async def run_pipeline(
        self,
        num_examples: int = 10,
    ) -> List[CodeExample]:
        """Run the complete pipeline"""
        print(f"Starting pipeline to generate {num_examples} examples...")
        
        # Stage 1: Research issues
        print("\n[Stage 1] Researching type checker issues...")
        issues = await self.research_agent.search_issues(max_issues=num_examples)
        print(f"Found {len(issues)} potential issues")
        
        validated_examples = []
        total_rounds = 0
        
        # Process each issue
        for i, issue in enumerate(issues, 1):
            print(f"\n[Issue {i}/{len(issues)}] Processing: {issue.title}")
            
            # Stage 2: Generate initial example
            print("  [Stage 2] Generating code example...")
            example = await self.generation_agent.generate_example(issue)
            
            # Stage 3 & 4: Validate and refine until good
            rounds = 0
            while rounds < self.max_refinement_rounds:
                rounds += 1
                total_rounds += 1
                
                print(f"  [Stage 3] Validation round {rounds}...")
                validation = self.validation_agent.validate_example(example)
                
                if validation.is_valid_syntax and validation.has_divergence:
                    print("  ✓ Example validated successfully!")
                    validated_examples.append(example)
                    break
                
                if rounds >= self.max_refinement_rounds:
                    print("  ✗ Max refinement rounds reached, skipping...")
                    break
                
                # Stage 4: Refine
                print("  [Stage 4] Refining example...")
                example = await self.refinement_agent.refine_example(
                    example, validation
                )
        
        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"Total refinement rounds: {total_rounds}")
        print(f"Successfully validated: {len(validated_examples)}/{num_examples}")
        print(f"{'='*60}")
        
        return validated_examples

# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Generate type checker divergence examples using multi-stage pipeline"
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
        help="Number of examples to generate (default: 10)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max refinement rounds per example (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="type_checker_examples.json",
        help="Output file for examples (default: type_checker_examples.json)",
    )
    
    args = parser.parse_args()
    
    # Get GitHub token
    token = os.environ.get("GITHUB_PAT")
    if not token:
        raise ValueError("Please set GITHUB_PAT environment variable")
    
    # Run pipeline
    pipeline = TypeCheckerDivergencePipeline(
        github_token=token,
        model=args.model,
        max_refinement_rounds=args.max_rounds,
    )
    
    examples = await pipeline.run_pipeline(num_examples=args.num_examples)
    
    # Save results
    output_data = [example.model_dump() for example in examples]
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Examples saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATED EXAMPLES:")
    print("="*60)
    for example in examples:
        print(f"\n{example.id} ({example.category})")
        print(f"References: {', '.join(example.issue_references)}")
        print(f"Reason: {example.reason}")
        print("\nCode:")
        print(example.code)
        print("-" * 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
