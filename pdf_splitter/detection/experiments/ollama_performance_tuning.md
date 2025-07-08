# Ollama Performance Tuning Guide

## Environment Variables for Performance

### 1. **OLLAMA_NUM_PARALLEL** (Default: 1)
Controls how many requests Ollama can process simultaneously.
```bash
export OLLAMA_NUM_PARALLEL=4  # Allow 4 parallel requests
```

### 2. **OLLAMA_MAX_LOADED_MODELS** (Default: 1)
Maximum number of models to keep loaded in memory.
```bash
export OLLAMA_MAX_LOADED_MODELS=3  # Keep 3 models in memory
```

### 3. **OLLAMA_MODELS_PATH**
Store models on faster storage (SSD vs HDD).
```bash
export OLLAMA_MODELS_PATH=/path/to/ssd/models
```

### 4. **OLLAMA_KEEP_ALIVE** (Default: 5m)
How long to keep models loaded after last use.
```bash
export OLLAMA_KEEP_ALIVE=30m  # Keep models loaded for 30 minutes
```

### 5. **OLLAMA_DEBUG**
Enable debug logging to identify bottlenecks.
```bash
export OLLAMA_DEBUG=1
```

## API-Level Optimizations

### 1. **Streaming Responses**
Use streaming for faster perceived response times:
```python
response = ollama.generate(
    model="model_name",
    prompt="prompt",
    stream=True  # Get tokens as they're generated
)
```

### 2. **Context Window Management**
Reuse context between calls for faster inference:
```python
# First call
response1 = ollama.generate(model="model", prompt="prompt1")
context = response1.get("context")

# Subsequent call with context
response2 = ollama.generate(
    model="model",
    prompt="prompt2",
    context=context  # Reuse previous context
)
```

### 3. **Batch Processing**
Process multiple prompts efficiently:
```python
# Instead of sequential calls
for prompt in prompts:
    response = ollama.generate(model="model", prompt=prompt)

# Consider parallel processing with threading
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(ollama.generate, model="model", prompt=p) for p in prompts]
    results = [f.result() for f in futures]
```

## Model-Specific Optimizations

### 1. **Quantization Levels**
- Q4_K_M: Good balance of speed and quality
- Q4_0: Faster but lower quality
- Q8_0: Slower but higher quality

### 2. **Model Size vs Speed**
Approximate inference speeds (CPU):
- 0.5-1B params: 50-100 tokens/sec
- 1-2B params: 20-50 tokens/sec
- 3-4B params: 10-20 tokens/sec
- 7-8B params: 5-10 tokens/sec

## System-Level Optimizations

### 1. **CPU Optimizations**
```bash
# Set CPU governor to performance mode
sudo cpupower frequency-set -g performance

# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. **Memory Optimization**
```bash
# Increase system file descriptor limits
ulimit -n 65536

# Adjust swap usage
sudo sysctl vm.swappiness=10
```

### 3. **Process Priority**
```bash
# Run Ollama with higher priority
sudo nice -n -10 ollama serve
```

## Testing Performance Settings

Create a test script to measure impact:
```python
import os
import time
import subprocess

# Test different settings
settings = [
    {},  # Default
    {"OLLAMA_NUM_PARALLEL": "4"},
    {"OLLAMA_MAX_LOADED_MODELS": "3"},
    {"OLLAMA_KEEP_ALIVE": "30m"},
    {"OLLAMA_NUM_PARALLEL": "4", "OLLAMA_MAX_LOADED_MODELS": "3"}
]

for config in settings:
    # Set environment variables
    env = os.environ.copy()
    env.update(config)

    # Restart Ollama with new settings
    subprocess.run(["ollama", "stop"], env=env)
    time.sleep(2)
    subprocess.Popen(["ollama", "serve"], env=env)
    time.sleep(5)

    # Run performance test
    # ... measure performance ...
```

## Recommended Production Settings

For the PDF boundary detection use case:
```bash
# Keep frequently used models in memory
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=30m

# Allow parallel processing if handling multiple PDFs
export OLLAMA_NUM_PARALLEL=2

# Store models on SSD if available
export OLLAMA_MODELS_PATH=/path/to/ssd/ollama/models
```
