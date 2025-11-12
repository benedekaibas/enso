import asyncio
import json
import os
import re
import shutil
import subprocess
import glob
from typing import Dict, List, Optional, Any

from pydantic_ai import Agent
from pydantic_ai.models.gpt4o import GPT4oModel


# ==============
# Config / Agent
# ==============

# Create an evaluation agent
eval_agent = Agent(
    model=GPT4oModel(),
    system_prompt=(
        "You are an expert in Python type systems and type checker behavior analysis. "
        "Your task is to determine which type checker is CORRECT when they disagree on code examples."
    ),
)


# ==============
# Helpers
# ==============

def find_latest_json() -> Optional[str]:
    """Find the most recent JSON file in dated folders matching 'output_*'."""
    dated_folders = glob.glob("output_*")
    if not dated_folders:
        return None

    latest_folder = max(dated_folders, key=os.path.getmtime)
    json_files = glob.glob(os.path.join(latest_folder, "*.json"))
    if not json_files:
        return None

    latest_json = max(json_files, key=os.path.getmtime)
    return latest_json


def parse_evaluation_result(text: str) -> Dict[str, Any]:
    """Parse the AI evaluation result into structured data."""
    lines = text.split("\n")
    evaluation: Dict[str, Any] = {}

    for line in lines:
        line = line.strip()
        if line.startswith("CORRECT_BEHAVIOR:"):
            evaluation["correct_behavior"] = line.split(":", 1)[1].strip()
        elif line.startswith("CORRECT_CHECKERS:"):
            checkers = line.split(":", 1)[1].strip().strip("[]")
            evaluation["correct_checkers"] = [c.strip() for c in checkers.split(",")] if checkers else []
        elif line.startswith("INCORRECT_CHECKERS:"):
            checkers = line.split(":", 1)[1].strip().strip("[]")
            evaluation["incorrect_checkers"] = [c.strip() for c in checkers.split(",")] if checkers else []
        elif line.startswith("REASONING:"):
            evaluation["reasoning"] = line.split(":", 1)[1].strip()
        elif line.startswith("TYPE_SYSTEM_ISSUE:"):
            evaluation["type_system_issue"] = line.split(":", 1)[1].strip()

    return evaluation


def extract_save_snippets(
    json_path: str, output_dir: str, code_key: str, extension: str
) -> List[str]:
    """Reads a JSON file, extracts code snippets, saves them to individual files."""
    created_files: List[str] = []

    try:
        with open(json_path, "r", encoding="utf-8") as fn:
            data = json.load(fn)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return created_files
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return created_files

    if not isinstance(data, list):
        print(f"Error: Expected JSON array at top level, got {type(data)}")
        return created_files

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    code_snippet_count = 0

    for index, item in enumerate(data):
        if isinstance(item, dict) and code_key in item:
            filename = os.path.join(output_dir, f"code_snippet_{index}{extension}")
            try:
                with open(filename, "w", encoding="utf-8") as output_file:
                    output_file.write(item[code_key])
                code_snippet_count += 1
                created_files.append(filename)
            except IOError as e:
                print(f"Error writing file: {filename}: {e}")

    print(f"🎉 Successfully extracted {code_snippet_count} code snippets and saved them to {output_dir}.")
    return created_files


def run_type_checkers(files_to_check: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Runs type checkers on code files."""
    type_checkers = [
        ("mypy", ["mypy"]),
        ("pyrefly", ["pyrefly", "check"]),
        ("zuban", ["zuban", "check"]),
        ("ty", ["ty", "check"]),
    ]

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    print("\n--- Starting Automated Type Checking ---")

    for filename in files_to_check:
        print(f"\n--- Processing File: {filename} ---")
        file_results: Dict[str, Dict[str, Any]] = {}

        for name, base_command in type_checkers:
            command = base_command + [filename]
            print(f"  -> Running {name}...")

            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                output = (result.stdout or "") + (result.stderr or "")
                status = "PASS" if result.returncode == 0 else "FAIL"

                file_results[name] = {
                    "status": status,
                    "return_code": result.returncode,
                    "output": output,
                }

                print(f"     Status: {status} (Exit Code: {result.returncode})")
                if status == "FAIL":
                    print("--- ERROR OUTPUT ---")
                    print(output.strip())
                    print("--------------------")

            except FileNotFoundError:
                file_results[name] = {
                    "status": "TOOL_MISSING",
                    "return_code": None,
                    "output": f"Error: {name} command not found.",
                }
                print(f"     Status: TOOL_MISSING (Is {name} installed and in PATH?)")

        all_results[filename] = file_results

    return all_results


async def evaluate_divergence(
    code: str, results: Dict[str, Any], expected_divergence: str = ""
) -> Dict[str, Any]:
    """Use pydantic-ai to evaluate which type checker is correct."""
    prompt = f"""
Analyze this Python code and type checker results to determine which type checker is CORRECT:

CODE:
{code}

TYPE CHECKER RESULTS:
{json.dumps(results, indent=2)}

EXPECTED DIVERGENCE (if provided):
{expected_divergence}

INSTRUCTIONS:

    Analyze the code for type correctness
    Determine which type checker behavior is correct (PASS or FAIL)
    Explain the reasoning based on Python typing specifications
    Identify any false positives or false negatives

Return your analysis in this format:

    CORRECT_BEHAVIOR: PASS/FAIL
    CORRECT_CHECKERS: [list of checker names that were correct]
    INCORRECT_CHECKERS: [list of checker names that were wrong]
    REASONING: [detailed explanation]
    TYPE_SYSTEM_ISSUE: [description of the underlying typing issue]
    """.strip()

    try:
        result = await eval_agent.run(prompt)
        content = getattr(result, "data", None) or getattr(result, "text", None) or str(result)
        return parse_evaluation_result(content)
    except Exception as e:
        return {"error": str(e)}


async def show_output() -> int:
    """Summarize the entire process: extraction, checking, and AI evaluation."""
    json_file_path = find_latest_json()
    if json_file_path is None:
        print("❌ No JSON files found. Run create_json.py first.")
        return 1

    json_folder = os.path.dirname(json_file_path)
    output_directory = os.path.join(json_folder, "extracted_python_snippets")

    print(f"📁 Using JSON file: {json_file_path}")
    print(f"📁 Output directory: {output_directory}")

    created_files = extract_save_snippets(
        json_path=json_file_path,
        output_dir=output_directory,
        code_key="code",
        extension=".py",
    )

    if not created_files:
        print("\nProcess aborted: No files were extracted to check.")
        return 1

    results = run_type_checkers(created_files)

    print("\n\n--- AI EVALUATION OF DIVERGENCES ---")
    evaluations: Dict[str, Dict[str, Any]] = {}

    for filename, file_results in results.items():
        print(f"\n🤖 Evaluating: {filename}")

        with open(filename, "r", encoding="utf-8") as f:
            code_content = f.read()

        expected_match = re.search(r"#\\s*EXPECTED:(.*?)#\\s*REASON:", code_content, re.DOTALL)
        expected_divergence = expected_match.group(1).strip() if expected_match else ""

        evaluation = await evaluate_divergence(code_content, file_results, expected_divergence)
        evaluations[filename] = evaluation

        print(f"   Correct behavior: {evaluation.get('correct_behavior', 'Unknown')}")
        print(f"   Correct checkers: {evaluation.get('correct_checkers', [])}")
        print(f"   Incorrect checkers: {evaluation.get('incorrect_checkers', [])}")
        if "reasoning" in evaluation:
            print(f"   Reasoning: {evaluation['reasoning'][:100]}...")

    print("\n\n--- FINAL SUMMARY WITH AI EVALUATION ---")
    total_failures = sum(
        1 for f, res in results.items() if any(r["status"] != "PASS" for r in res.values())
    )

    eval_file = os.path.join(json_folder, "ai_evaluations.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)
    print(f"💾 AI evaluations saved to: {eval_file}")

    print(f"\nTotal files checked: {len(created_files)}")
    print(f"Total files with check failures: {total_failures}")
    return 1 if total_failures else 0


if __name__ == "__main__":
    final_status = asyncio.run(show_output())
    print(f"\nProcess finished with system exit code {final_status}")

