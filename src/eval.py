import json
import os
import re
import shutil
import subprocess
import glob
from typing import Dict, List, Optional, Any

def find_latest_snippets_dir() -> Optional[str]:
    candidates = glob.glob("output_*/extracted_python_snippets")
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return latest

def analyze_code_behavior(code: str) -> Dict[str, Any]:
    """Analyze the actual Python code to determine expected type behavior."""
    analysis = {
        "correct_behavior": "UNKNOWN",
        "reasoning": "",
        "type_system_issue": ""
    }
    
    # Analyze code patterns to determine expected behavior
    code_lower = code.lower()
    
    # Check for obvious type errors in the code
    if "newtype" in code_lower and "list[int]" in code_lower:
        # NewType with List covariance issue
        analysis.update({
            "correct_behavior": "FAIL",
            "reasoning": "NewType creates nominal types; List[int] should not be assignable to List[NewType] due to invariance",
            "type_system_issue": "NewType container variance"
        })
    elif "protocol" in code_lower and "default" in code_lower:
        # Protocol with default argument differences
        analysis.update({
            "correct_behavior": "FAIL", 
            "reasoning": "Protocol implementations should match method signatures exactly, including default values",
            "type_system_issue": "Protocol default argument compatibility"
        })
    elif "typeguard" in code_lower and "append" in code_lower:
        # TypeGuard with container modification
        analysis.update({
            "correct_behavior": "FAIL",
            "reasoning": "TypeGuard may not properly narrow generic containers for mutation operations",
            "type_system_issue": "TypeGuard container narrowing"
        })
    elif "final" in code_lower and "property" in code_lower:
        # Final attribute override with property
        analysis.update({
            "correct_behavior": "FAIL",
            "reasoning": "Final attributes cannot be overridden, even with properties",
            "type_system_issue": "Final attribute inheritance"
        })
    elif "overload" in code_lower and "literal" in code_lower:
        # Overload with literal types
        analysis.update({
            "correct_behavior": "PASS",
            "reasoning": "Overload resolution with Literal types should work correctly",
            "type_system_issue": "Overload literal discrimination"
        })
    elif "typeddict" in code_lower and "required" in code_lower:
        # TypedDict with Required/NotRequired
        analysis.update({
            "correct_behavior": "PASS", 
            "reasoning": "Mixed total/not-total TypedDict inheritance should be handled consistently",
            "type_system_issue": "TypedDict inheritance semantics"
        })
    
    return analysis

def run_type_checkers(files_to_check: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    type_checkers = [
        ("mypy", ["mypy"]),
        ("pyrefly", ["pyrefly", "check"]),
        ("zuban", ["zuban", "check"]),
        ("ty", ["ty", "check"]),
    ]

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    print("\nRunning type checkers")

    for filename in sorted(files_to_check):
        print(f"\nProcessing {os.path.basename(filename)}")
        file_results: Dict[str, Dict[str, Any]] = {}
        for name, base_command in type_checkers:
            cmd = base_command + [filename]
            print(f"  -> {name}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                output = (result.stdout or "") + (result.stderr or "")
                status = "PASS" if result.returncode == 0 else "FAIL"
                file_results[name] = {
                    "status": status,
                    "return_code": result.returncode,
                    "output": output,
                }
                print(f"     Status: {status}")
                if status == "FAIL" and output.strip():
                    print(f"     Error: {output.strip()[:100]}...")
            except FileNotFoundError:
                file_results[name] = {
                    "status": "TOOL_MISSING",
                    "return_code": None,
                    "output": f"{name} not found",
                }
                print(f"     {name} not found")
        all_results[filename] = file_results
    return all_results

def evaluate_type_checkers(file_results: Dict[str, Dict[str, Any]], code_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate which type checkers are correct based on code analysis."""
    expected_behavior = code_analysis["correct_behavior"]
    
    if expected_behavior == "UNKNOWN":
        return {
            "correct_behavior": "UNKNOWN",
            "correct_checkers": [],
            "incorrect_checkers": [],
            "reasoning": "Unable to determine expected behavior from code analysis",
            "type_system_issue": code_analysis["type_system_issue"]
        }
    
    correct_checkers = []
    incorrect_checkers = []
    
    for checker_name, result in file_results.items():
        if result["status"] == expected_behavior:
            correct_checkers.append(checker_name)
        elif result["status"] in ["PASS", "FAIL"]:  # Only count actual results, not missing tools
            incorrect_checkers.append(checker_name)
    
    return {
        "correct_behavior": expected_behavior,
        "correct_checkers": correct_checkers,
        "incorrect_checkers": incorrect_checkers,
        "reasoning": code_analysis["reasoning"],
        "type_system_issue": code_analysis["type_system_issue"]
    }

def main() -> int:
    snippets_dir = find_latest_snippets_dir()
    if not snippets_dir:
        print("No extracted_python_snippets directory found under output_*")
        return 1

    py_files = sorted(glob.glob(os.path.join(snippets_dir, "*.py")))
    if not py_files:
        print(f"No .py files found in {snippets_dir}")
        return 1

    print(f"Using snippets directory: {snippets_dir}")
    results = run_type_checkers(py_files)

    evaluations: Dict[str, Dict[str, Any]] = {}
    print("\n--- CODE ANALYSIS AND EVALUATION ---")
    
    for filename, file_results in results.items():
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
        
        # Analyze the actual code to determine expected behavior
        code_analysis = analyze_code_behavior(code)
        evaluation = evaluate_type_checkers(file_results, code_analysis)
        evaluations[filename] = evaluation
        
        print(f"\n{os.path.basename(filename)}:")
        print(f"  Expected: {evaluation['correct_behavior']}")
        print(f"  Correct: {evaluation['correct_checkers']}")
        print(f"  Incorrect: {evaluation['incorrect_checkers']}")
        print(f"  Issue: {evaluation['type_system_issue']}")

    # Save results
    parent = os.path.dirname(snippets_dir)
    eval_file = os.path.join(parent, "code_based_evaluations.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)
    print(f"\nSaved code-based evaluations to {eval_file}")

    # Summary statistics
    total_files = len(py_files)
    files_with_known_behavior = sum(1 for e in evaluations.values() if e["correct_behavior"] != "UNKNOWN")
    consistent_checkers = sum(1 for e in evaluations.values() if len(e["correct_checkers"]) >= 3)
    
    print("\n--- SUMMARY ---")
    print(f"Total files: {total_files}")
    print(f"Files with determinable behavior: {files_with_known_behavior}")
    print(f"Files with consistent checker behavior: {consistent_checkers}")
    
    return 0

if __name__ == "__main__":
    exit(main())

