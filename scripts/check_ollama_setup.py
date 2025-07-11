#!/usr/bin/env python3
"""Check and configure Ollama for the PDF splitter."""

import os
import subprocess
import time
import requests
import sys


def check_ollama_running():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_model_available(model_name="gemma2:2b"):
    """Check if the required model is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [m['name'] for m in models.get('models', [])]
            return any(model_name in name for name in model_names)
    except:
        return False


def start_ollama():
    """Try to start Ollama service."""
    print("Attempting to start Ollama...")
    try:
        # Try systemctl first (for systemd systems)
        result = subprocess.run(
            ["systemctl", "start", "ollama"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✓ Ollama service started via systemctl")
            return True
    except:
        pass
    
    try:
        # Try ollama serve command
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓ Started Ollama serve process")
        return True
    except:
        print("✗ Failed to start Ollama")
        return False


def pull_model(model_name="gemma2:2b"):
    """Pull the required model."""
    print(f"Pulling {model_name} model...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Successfully pulled {model_name}")
            return True
        else:
            print(f"✗ Failed to pull model: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error pulling model: {e}")
        return False


def test_model(model_name="gemma2:2b"):
    """Test the model with a simple query."""
    print(f"Testing {model_name} model...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Is this a new document? Answer with just yes or no.",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 10
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Model test successful. Response: {result.get('response', '')[:50]}...")
            return True
        else:
            print(f"✗ Model test failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Model test error: {e}")
        return False


def update_env_file():
    """Update .env file with correct Ollama settings."""
    env_path = ".env"
    
    settings = {
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "gemma2:2b",
        "OLLAMA_TIMEOUT": "120",
        "LLM_PROVIDER": "ollama"
    }
    
    # Read existing .env
    existing_lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            existing_lines = f.readlines()
    
    # Update or add settings
    updated = False
    new_lines = []
    
    for line in existing_lines:
        key = line.split('=')[0].strip() if '=' in line else None
        if key in settings:
            new_lines.append(f"{key}={settings[key]}\n")
            del settings[key]
            updated = True
        else:
            new_lines.append(line)
    
    # Add any remaining settings
    for key, value in settings.items():
        new_lines.append(f"{key}={value}\n")
        updated = True
    
    if updated:
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        print("✓ Updated .env file with Ollama settings")
    
    return True


def main():
    """Main setup routine."""
    print("PDF Splitter - Ollama Setup Check")
    print("=" * 50)
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True)
        if result.returncode != 0:
            print("✗ Ollama is not installed")
            print("\nTo install Ollama:")
            print("  curl -fsSL https://ollama.ai/install.sh | sh")
            return False
    except:
        print("✗ Could not check Ollama installation")
        return False
    
    print("✓ Ollama is installed")
    
    # Check if running
    if not check_ollama_running():
        print("✗ Ollama is not running")
        if start_ollama():
            time.sleep(5)  # Give it time to start
            if not check_ollama_running():
                print("✗ Ollama failed to start properly")
                return False
        else:
            return False
    else:
        print("✓ Ollama is running")
    
    # Check model availability
    model = "gemma2:2b"  # Using smaller, faster model
    if not check_model_available(model):
        print(f"✗ Model {model} not available")
        if not pull_model(model):
            # Try alternative model
            model = "gemma:2b"
            print(f"Trying alternative model: {model}")
            if not pull_model(model):
                print("✗ Could not pull any suitable model")
                return False
    else:
        print(f"✓ Model {model} is available")
    
    # Test model
    if not test_model(model):
        print("✗ Model test failed")
        print("\nTroubleshooting:")
        print("1. Check Ollama logs: journalctl -u ollama -f")
        print("2. Try restarting: systemctl restart ollama")
        print("3. Check port 11434 is not blocked")
        return False
    
    # Update .env file
    update_env_file()
    
    print("\n" + "=" * 50)
    print("✓ Ollama setup complete!")
    print(f"  Model: {model}")
    print("  Host: http://localhost:11434")
    print("  Timeout: 120 seconds")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)