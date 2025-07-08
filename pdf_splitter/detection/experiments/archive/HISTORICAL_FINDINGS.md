# Historical Experiment Findings

This document consolidates key learnings from all previous LLM boundary detection experiments conducted before the implementation of the systematic prompt engineering framework.

## Key Discoveries

### 1. The Fundamental Challenge (July 2025)
Initial experiments revealed that many models were simply marking every page transition as a boundary, achieving "100% recall" by classifying everything as "Different Documents". This was not real boundary detection but a bias in the prompts.

### 2. Model Behavior Insights

#### Models That Follow Instructions
- **phi4-mini:3.8b**: Generally good instruction following, fast performance (~2s/boundary)
- **gemma3:latest**: Consistent behavior, good at following simple instructions
- **phi3:mini**: Reliable but slightly slower

#### Problematic Models
- **qwen3:8b**: Poor instruction following despite good conceptual understanding
- **qwen3:1.7b**: Inconsistent behavior across runs
- **deepseek-r1 variants**: Complete failure on boundary detection tasks

### 3. Prompt Engineering Learnings

#### What Doesn't Work
- Complex prompts with multiple instructions
- Asking for explanations along with classification
- Vague instructions like "analyze these pages"
- Two-pass verification (minimal benefit for 2x processing time)

#### What Works
- Simple, direct instructions
- Single-character outputs (S/D)
- Conservative bias ("assume same unless clear evidence")
- Structured decision formats

### 4. Performance Benchmarks

Best results achieved (before systematic framework):
- **Speed**: 2.04s per boundary (phi4-mini:3.8b)
- **Accuracy**: F1=0.667 (qwen3:8b, but poor instruction following)
- **Best Balance**: phi4-mini:3.8b with F1=0.526, 100% recall

### 5. Multi-Modal Experiments
Brief testing with image-based detection showed:
- 4x slower than text-based approaches
- Lower accuracy on document boundaries
- Not worth pursuing for primary detection

### 6. Critical Insights

1. **False positives are acceptable** - easier to merge than split documents
2. **100% recall is achievable** - but often at the cost of many false positives
3. **Simple approaches work best** - complexity doesn't improve accuracy
4. **Model size doesn't guarantee performance** - smaller models often outperform larger ones
5. **Instruction following varies widely** - test each model's ability to follow simple instructions first

### 7. Failed Approaches

- **Type-first detection**: Classifying document type before boundary detection added complexity without benefit
- **Chain-of-thought prompting**: Increased latency without improving accuracy
- **Confidence scoring**: Models struggled to calibrate confidence accurately
- **Multi-signal approaches**: Premature optimization before getting LLM detection working

## Conclusion

These experiments led to the development of the systematic prompt engineering framework, which focuses on:
1. Progressive difficulty testing (easy → medium → hard)
2. Simple, testable prompt formats
3. Automated testing across multiple models
4. Clear metrics and evaluation criteria

The key lesson: Start simple, test systematically, and let data guide optimization.
