import subprocess
import json
import re
import sys

def extract_code_blocks(text):
    """Extract Python code blocks and metadata from the text."""
    code_blocks = []
    
    # Split by code blocks
    blocks = re.split(r'```python', text)
    
    for block in blocks:
        if '```' in block:
            code_content = block.split('```')[0].strip()
            if code_content:
                # Extract metadata from comments
                id_match = re.search(r'# id:\s*([^\n]+)', code_content)
                expected_match = re.search(r'# EXPECTED:(.*?)# REASON:', code_content, re.DOTALL)
                reason_match = re.search(r'# REASON:\s*([^\n]+)', code_content)
                
                code_blocks.append({
                    "id": id_match.group(1).strip() if id_match else "unknown",
                    "code": code_content,
                    "expected": expected_match.group(1).strip() if expected_match else "",
                    "reason": reason_match.group(1).strip() if reason_match else ""
                })
    
    return code_blocks

def main():
    # Read from stdin (terminal output)
    terminal_output = sys.stdin.read()
    
    code_blocks = extract_code_blocks(terminal_output)
    
    # Output as JSON to stdout with the correct structure
    data = [
        {
            "title": f"Type Checker Divergence {i+1}",
            "description": f"Code snippet demonstrating type checker divergence: {block.get('id', 'unknown')}",
            "code": block["code"]
        }
        for i, block in enumerate(code_blocks)
    ]
    
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
