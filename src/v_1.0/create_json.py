import subprocess
import json
import re
import sys
import os
from datetime import datetime

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
    
    # Create dated folder
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"output_{today}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output as JSON to file with timestamp
    timestamp = datetime.now().strftime("%H-%M-%S")
    output_file = f"{output_dir}/code_examples_{timestamp}.json"
    
    data = [
        {
            "title": f"Type Checker Divergence {i+1}",
            "description": f"Code snippet demonstrating type checker divergence: {block.get('id', 'unknown')}",
            "code": block["code"]
        }
        for i, block in enumerate(code_blocks)
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(data)} code examples to: {output_file}")
    
    # Also print the file path so automation.py can use it
    print(f"OUTPUT_FILE:{output_file}")

if __name__ == "__main__":
    main()
