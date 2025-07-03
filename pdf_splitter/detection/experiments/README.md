# LLM Detection Experiments

This module provides a framework for experimenting with different LLM models and strategies for document boundary detection.

## Structure

```
experiments/
├── configs/           # Experiment configurations (auto-generated)
├── results/          # JSON results from experiments
├── analysis/         # Jupyter notebooks for analysis (create as needed)
├── prompts/          # Prompt templates
├── experiment_runner.py    # Core experiment framework
└── run_experiments.py      # CLI for running experiments
```

## Running Experiments

### Basic Usage

```bash
# Run with default settings (Llama3, context overlap strategy)
python -m pdf_splitter.detection.experiments.run_experiments

# Test specific models
python -m pdf_splitter.detection.experiments.run_experiments \
    --models llama3:8b-instruct-q5_K_M gemma3:latest phi4-mini:3.8b \
    --strategies context_overlap chain_of_thought

# Compare existing results
python -m pdf_splitter.detection.experiments.run_experiments \
    --compare-only \
    --models llama3:8b-instruct-q5_K_M gemma3:latest
```

### Available Strategies

1. **context_overlap**: Sliding window with configurable overlap
   - Tests 20%, 30%, and 40% overlap
   - Good baseline strategy

2. **type_first**: Classify document type, then detect boundaries
   - Two-pass approach
   - May improve accuracy for diverse document sets

3. **chain_of_thought**: Step-by-step reasoning
   - Higher latency but potentially more accurate
   - Good for understanding model reasoning

4. **multi_signal**: Combine multiple detection signals
   - Placeholder for integration with visual/heuristic detectors

## Adding New Experiments

### 1. Create a New Strategy

Edit `experiment_runner.py` and add a new method:

```python
def _run_my_strategy(self, config, pages, result):
    # Your strategy implementation
    predictions = []
    # ...
    return predictions
```

### 2. Create a New Prompt Template

Add a `.txt` file to the `prompts/` directory:

```
prompts/my_template.txt
```

Then reference it in your experiment config:
```python
config = ExperimentConfig(
    name="my_experiment",
    model="llama3:latest",
    strategy="my_strategy",
    prompt_template="my_template"
)
```

### 3. Analyze Results

Results are saved as JSON in the `results/` directory. Each file contains:
- Configuration used
- Predicted vs actual boundaries
- Performance metrics (precision, recall, F1)
- Timing information
- Model responses and errors

## Performance Targets

- **Accuracy**: > 95% F1 score
- **Latency**: < 2 seconds per boundary check
- **Consistency**: Low variance across runs

## Tips for Experimentation

1. **Start Simple**: Begin with context_overlap strategy as baseline
2. **Test Incrementally**: Change one variable at a time
3. **Monitor Errors**: Check the errors field in results
4. **Validate Prompts**: Test prompts manually with Ollama first
5. **Consider Edge Cases**: First/last pages, single-page documents

## Ollama Setup

Ensure Ollama is running:
```bash
ollama serve
```

Pull required models:
```bash
ollama pull llama3:8b-instruct-q5_K_M
ollama pull gemma3:latest
ollama pull phi4-mini:3.8b
```
