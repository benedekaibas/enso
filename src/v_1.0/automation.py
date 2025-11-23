import json
import os
import subprocess
import shutil 
import glob
from datetime import datetime

def find_latest_json():
    """Find the most recent JSON file in dated folders."""
    # Look for folders like output_2024-12-19
    dated_folders = glob.glob("output_*")
    if not dated_folders:
        return None
    
    # Get the most recent folder
    latest_folder = max(dated_folders, key=os.path.getmtime)
    
    # Find all JSON files in that folder
    json_files = glob.glob(os.path.join(latest_folder, "*.json"))
    if not json_files:
        return None
    
    # Get the most recent JSON file
    latest_json = max(json_files, key=os.path.getmtime)
    return latest_json

# Auto-detect the latest JSON file
JSON_FILE_PATH = find_latest_json()
if JSON_FILE_PATH is None:
    print("❌ No JSON files found. Run create_json.py first.")
    exit(1)

# Get the folder containing the JSON file
JSON_FOLDER = os.path.dirname(JSON_FILE_PATH)
OUTPUT_DIRECTORY = os.path.join(JSON_FOLDER, 'extracted_python_snippets')
CODE_KEY = 'code'
FILE_EXTENSION = '.py'

def extractSaveSnippets(jsonPath: str, outputDir: str, codeKey: str, extension: str) -> list:
    """Reads a JSON file, extracts code snippets, saves them to individual files."""
    created_files = []
    
    try:
        with open(jsonPath, "r", encoding='utf-8') as fn:
            data = json.load(fn) 
    except FileNotFoundError:
        print(f"Error: JSON file not found at {jsonPath}")
        return created_files
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {jsonPath}")
        return created_files
    
    # Direct array handling for the new JSON format
    if isinstance(data, list):
        codeSnippets = data
    else: 
        print(f"Error: Expected JSON array at top level, got {type(data)}")
        return created_files

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir) 
    os.makedirs(outputDir)
    
    codeSnippetCount = 0

    for index, item in enumerate(codeSnippets):
        if isinstance(item, dict) and codeKey in item:
            filename = os.path.join(outputDir, f"code_snippet_{index}{extension}")
            try:
                with open(filename, "w", encoding='utf-8') as outputFile:
                    outputFile.write(item[codeKey])
                codeSnippetCount += 1
                created_files.append(filename)
            except IOError as e:
                print(f"Error writing file: {filename}: {e}")

    print(f"🎉 Successfully extracted {codeSnippetCount} code snippets and saved them to {outputDir}.")
    return created_files

def runTypeChecker(files_to_check: list):
    """
    Runs a suite of type checkers on a list of specified code files.
    """
    typeCheckers = [
        ("mypy", ["mypy"]), 
        ("pyrefly", ["pyrefly", "check"]),
        ("zuban", ["zuban", "check"]),
        ("ty", ["ty", "check"]),
    ]
    
    all_results = {} 

    print("\n--- Starting Automated Type Checking ---")
    
    for filename in files_to_check:
        print(f"\n--- Processing File: {filename} ---")
        file_results = {}
        
        for name, base_command in typeCheckers:
            command = base_command + [filename] 
            print(f"  -> Running {name}...")
            
            try:
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True, 
                    check=False
                )
                
                output = result.stdout + result.stderr
                
                # SIMPLIFIED: If return code is non-zero, it's a FAIL
                # Type checkers only return non-zero for actual errors, not warnings
                status = "PASS" if result.returncode == 0 else "FAIL"
                
                file_results[name] = {
                    "status": status,
                    "return_code": result.returncode,
                    "output": output
                }
                
                print(f"     Status: {status} (Exit Code: {result.returncode})")
                if status == "FAIL":
                    print("--- ERROR OUTPUT ---")
                    print(output.strip())
                    print("--------------------")
                
            except FileNotFoundError:
                file_results[name] = {"status": "TOOL_MISSING", "output": f"Error: {name} command not found."}
                print(f"     Status: TOOL_MISSING (Is {name} installed and in PATH?)")
                
        all_results[filename] = file_results
    
    return all_results

def showOutput():
    """
    Summarize the entire process: extraction, checking, and summary.
    """
    print(f"📁 Using JSON file: {JSON_FILE_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIRECTORY}")
    
    # Extract and save all code snippets
    created_files = extractSaveSnippets(
        jsonPath=JSON_FILE_PATH,
        outputDir=OUTPUT_DIRECTORY,
        codeKey=CODE_KEY,
        extension=FILE_EXTENSION
    )
    
    if not created_files:
        print("\nProcess aborted: No files were extracted to check.")
        return 1
        
    results = runTypeChecker(created_files)

    print("\n\n--- FINAL SUMMARY ---")
    overall_exit_code = 0
    total_failures = 0
    
    for filename, file_results in results.items():
        print(f"\nResults for {filename}:")
        file_failed = False
        for checker_name, data in file_results.items():
            status = data['status']
            print(f"  - {checker_name}: {status}")
            if status != "PASS":
                file_failed = True
                
        if file_failed:
            total_failures += 1
            overall_exit_code = 1
            
    print(f"\nTotal files checked: {len(created_files)}")
    print(f"Total files with check failures: {total_failures}")
    
    # shutil.rmtree(OUTPUT_DIRECTORY) # files for inspection
    # print(f"\nCleanup complete. Removed directory: {OUTPUT_DIRECTORY}")

    return overall_exit_code

if __name__ == '__main__':
    final_status = showOutput()
    print(f"\nProcess finished with system exit code {final_status}")
