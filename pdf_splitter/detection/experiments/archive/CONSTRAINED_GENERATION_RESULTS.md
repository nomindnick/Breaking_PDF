# Constrained Generation Test Results

## Date: January 5, 2025

### Executive Summary

We tested three approaches to constrained generation to ensure models only output valid responses:
1. **Regex constraint** (single token S/D): Best F1=0.435, fastest (2.3s)
2. **JSON structure**: Failed (F1=0.000), all predictions were "Same"
3. **XML completion**: Mixed results (F1=0.352 for phi4, 0.000 for gemma3)

**Conclusion**: Constrained generation underperforms compared to the optimal prompts (F1=0.700), but offers faster response times.

### Detailed Results

| Model | Approach | F1 Score | Accuracy | Avg Time | Prediction Distribution |
|-------|----------|----------|----------|----------|------------------------|
| phi4-mini:3.8b | regex | 0.435 | 50.00% | 2.313s | Same=13, Different=13 |
| phi4-mini:3.8b | json | 0.000 | 61.54% | 4.685s | Same=26, Different=0 |
| phi4-mini:3.8b | xml | 0.352 | 61.54% | 3.498s | Same=22, Different=4 |
| gemma3:latest | regex | 0.342 | 38.46% | 2.566s | Same=18, Different=8 |
| gemma3:latest | json | 0.000 | 61.54% | 4.617s | Same=26, Different=0 |
| gemma3:latest | xml | 0.000 | 61.54% | 3.141s | Same=26, Different=0 |

### Key Observations

1. **Speed vs Accuracy Trade-off**:
   - Constrained generation is ~10-15x faster than optimal prompts
   - But F1 scores are significantly lower (0.435 vs 0.700)

2. **Conservative Bias**:
   - JSON and XML approaches predict almost all cases as "Same"
   - Only regex approach shows balanced predictions

3. **Implementation Challenges**:
   - Ollama doesn't support true logit-level constraints
   - We used stop tokens and post-processing as workarounds
   - This likely contributes to lower performance

### Comparison with Previous Results

| Approach | Best F1 | Response Time |
|----------|---------|---------------|
| gemma3_optimal | 0.700 | ~33s |
| E1_cod_reasoning | 0.500 | ~7s |
| Constrained regex | 0.435 | ~2.3s |
| D1_conservative_few_shot | 0.400 | ~2s |

### Technical Details

The constrained generation approaches tested:

1. **Regex Constraint**:
   - Simple prompt ending with "Decision:"
   - Single token output with stop tokens
   - Post-processing to extract S/D

2. **JSON Structure**:
   - Prompt requesting JSON with decision/confidence/reasoning
   - Stop at newlines or code blocks
   - JSON parsing with fallback

3. **XML Completion**:
   - Prompt with `<decision>` tag to complete
   - Stop at closing tag
   - Extract SAME/DIFFERENT from response

### Recommendations

1. **For Production Use**:
   - If speed is critical: Use constrained regex approach (2.3s, F1=0.435)
   - If accuracy is critical: Use gemma3_optimal prompt (33s, F1=0.700)
   - For balance: Consider E1_cod_reasoning (7s, F1=0.500)

2. **Future Improvements**:
   - Test with models that support true logit constraints
   - Experiment with better prompt engineering for constrained formats
   - Consider ensemble approach: fast constrained filter + slow accurate verification

3. **Not Recommended**:
   - JSON/XML constrained approaches due to poor performance
   - Conservative bias makes them unsuitable for boundary detection

### Lessons Learned

1. Simpler constraints (single token) work better than complex structures
2. Without true logit-level constraints, performance suffers
3. The optimal prompts' superior performance justifies their longer response time
4. Constrained generation might work better with different model architectures
