"""This file contains a code that makes automated process of feeding code examples into type checkers."""
import json
import os
import subprocess
import shutil 


JSON_FILE_PATH = 'data_two.json'
OUTPUT_DIRECTORY = 'extracted_python_snippets_two'
CODE_KEY = 'code'
OUTER_KEY = 'code_examples' # TODO: check if json outer list is named code_examples or we have to configure this variable
FILE_EXTENSION = '.py'



def extractSaveSnippets(jsonPath: str, outputDir: str, outerKey: str, codeKey: str, extension: str) -> list:
    """
    Reads a JSON file, extracts code snippets, saves them to individual files, 
    and returns a list of the created file paths.
    """
    created_files = []
    
    # load the json and receive its data
    try:
        with open(jsonPath, "r", encoding='utf-8') as fn:
            data = json.load(fn) 
    except FileNotFoundError:
        print(f"Error: JSON file not found at {jsonPath}")
        return created_files
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {jsonPath}")
        return created_files
    
    # determine if json has dicts containing lists in the inner part
    # the json is created by pydantic-ai, so the correct prompt is important!
    if isinstance(data, dict):
        codeSnippets = data.get(outerKey)
    elif isinstance(data, list):
        codeSnippets = data
    else: 
        print(f"Error: Unexpected top-level JSON type: {type(data)}. Expected dict or list.")
        return created_files

    # check if the pydantic generated json file's format is correct or not
    if not isinstance(codeSnippets, list):
        print(f"Error: Could not find a valid list of snippets. Check if the key '{outerKey}' is correct.")
        return created_files

    # make sure I have an existing directory for the code snippet outputs
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir) 
    os.makedirs(outputDir)
    
    codeSnippetCount = 0

    # iterate over the code snippets and give them to the output .py files 
    for index, item in enumerate(codeSnippets):
        if isinstance(item, dict) and codeKey in item:
            snippets = item[codeKey]
            fileId = index
            filename = os.path.join(outputDir, f"code_snippet_{fileId}{extension}")

            try:
                with open(filename, "w", encoding='utf-8') as outputFile:
                    outputFile.write(snippets)
                
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
        ("pyright", ["pyright"]),
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
                    check=False # checker reports error, but I do not want the script to crash
                )
                
                status = "PASS" if result.returncode == 0 else "FAIL"
                
                # Check for common failure messages in the output
                output = result.stdout + result.stderr # combine both stdout and stderr since they are displaying important err messages
                
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
    
    # Extract and save all code snippets
    created_files = extractSaveSnippets(
        jsonPath=JSON_FILE_PATH,
        outputDir=OUTPUT_DIRECTORY,
        outerKey=OUTER_KEY,
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
    # sys.exit(final_status) # reflect the type checking results
