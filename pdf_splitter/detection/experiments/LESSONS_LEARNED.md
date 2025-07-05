# Lessons Learned from Full Performance Testing

## Date: January 4, 2025

### 1. Chain-of-Draft (CoD) is Promising
- **E1_cod_reasoning** achieved the best F1 score (0.500)
- **E2_cod_minimal** showed good balance with simpler format
- Structured reasoning helps models make better decisions
- Trade-off: 3-4x slower than baseline prompts (5-7s vs 2s)

### 2. Testing Infrastructure Works Well
- Successfully tested 312 prompt/model/case combinations
- Automated evaluation with precision/recall/F1 metrics
- Progressive difficulty testing reveals performance degradation
- ~13 minutes for comprehensive test is reasonable

### 3. Critical Issues to Address

#### Immediate Fixes Needed:
1. **Naming Mismatch**: `phi_optimal` vs `phi4_optimal.txt`
   - Simple fix but prevented testing our best prompts
   - Always verify file loading logic matches actual filenames

2. **Response Parsing**: A1_asymmetric returns all "Unknown"
   - Need robust parsing that handles various response formats
   - Consider fallback parsing strategies

#### Performance Issues:
1. **Low F1 Scores**: Best is only 0.500
   - Dataset imbalance affects metrics
   - Models have conservative bias (prefer "Same")
   - Need better few-shot examples

2. **Accuracy vs Recall Trade-off**:
   - High recall (100%) comes with low precision (33%)
   - High accuracy (61%) comes with poor recall (30%)
   - Need to find better balance

### 4. Model Insights
- **Gemma3 and Phi4 perform similarly** overall
- Both struggle with harder test cases (33% on difficulty 9-10)
- Model-specific formatting likely important (untested)
- Smaller models can reason but need careful prompting

### 5. Prompt Engineering Insights

#### What Works:
- ✅ Structured output formats (XML, CoD steps)
- ✅ Clear decision criteria
- ✅ Forcing single-token or structured responses
- ✅ Progressive reasoning (even if minimal)

#### What Doesn't:
- ❌ Overly conservative biasing ("assume same unless certain")
- ❌ Single-character outputs without structure (A1)
- ❌ Prompts without examples or clear format

### 6. Dataset Observations
- **Imbalanced**: More "Same" cases than "Different"
- **Difficulty progression works**: Clear performance degradation
- **Edge cases are hard**: Only 33% accuracy on level 9-10
- Need more diverse test cases at each level

### 7. Next Experimental Priorities

1. **Test the Optimal Prompts**: Fix naming and rerun
2. **Implement Constrained Generation**: Force valid outputs
3. **Balance Dataset**: Equal Same/Different distribution
4. **Refine CoD Prompts**: Reduce false positives while maintaining recall
5. **Test on Real PDFs**: Validate synthetic results

### 8. Surprising Findings
- 📊 **E2_cod_minimal performed nearly as well as E1_cod_reasoning** despite being much simpler
- 🔄 **100% recall is achievable** but comes at huge precision cost
- ⏱️ **Response time scales with output length** as expected
- 🎯 **F1 score is more informative than accuracy** for imbalanced data

### 9. Technical Recommendations

#### For Production:
- Use E1_cod_reasoning for high-recall needs
- Use D1_conservative_few_shot for high-precision needs
- Consider ensemble approach combining both
- Implement confidence thresholds

#### For Further Research:
- Test impact of temperature settings
- Try different few-shot example sets
- Experiment with prompt length vs performance
- Test smaller/larger models

### 10. Key Takeaway
**The framework is solid, but prompt optimization is crucial.** The difference between 0% F1 (A1_asymmetric) and 50% F1 (E1_cod_reasoning) shows the massive impact of prompt engineering. The untested optimal prompts likely hold even more potential.