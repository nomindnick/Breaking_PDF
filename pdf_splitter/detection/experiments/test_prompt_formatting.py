#!/usr/bin/env python3
"""Quick test to verify prompt formatting works correctly."""

import json
from pathlib import Path
import requests

# Load prompts
prompts_dir = Path("pdf_splitter/detection/experiments/prompts")

# Test case
test_case = {
    "page1_bottom": "...and we thank you for your business.\nSincerely,\nJohn Smith",
    "page2_top": "INVOICE #12345\nDate: March 1, 2024\nCustomer: ABC Corp"
}

# Load and test phi4_optimal
phi4_path = prompts_dir / "phi4_optimal.txt"
if phi4_path.exists():
    print("PHI4 OPTIMAL PROMPT:")
    print("-" * 60)
    template = phi4_path.read_text()
    formatted = template.format(**test_case)
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
    print("-" * 60)
    
    # Test with Ollama
    print("\nTesting with phi4-mini:3.8b...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi4-mini:3.8b",
                "prompt": formatted,
                "temperature": 0.0,
                "options": {
                    "num_predict": 200,
                    "stop": ["<|im_end|>"]
                },
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("Response:", result.get("response", "")[:200])
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Error:", e)

print("\n" + "="*60 + "\n")

# Load and test gemma3_optimal
gemma_path = prompts_dir / "gemma3_optimal.txt"
if gemma_path.exists():
    print("GEMMA3 OPTIMAL PROMPT:")
    print("-" * 60)
    template = gemma_path.read_text()
    formatted = template.format(**test_case)
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
    print("-" * 60)
    
    # Test with Ollama
    print("\nTesting with gemma3:latest...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:latest",
                "prompt": formatted,
                "temperature": 0.0,
                "options": {
                    "num_predict": 200,
                    "stop": ["<end_of_turn>"]
                },
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("Response:", result.get("response", "")[:200])
        else:
            print("Error:", response.status_code)
    except Exception as e:
        print("Error:", e)