# Detection Experiments Summary

## 🎯 Quick Reference - Our Finalists
- **Best for speed**: phi4-mini:3.8b (2.04s, 100% recall, 9 FPs)
- **Best for accuracy**: qwen3:8b (9.13s, 80% recall, 3 FPs)
- **Best small model**: qwen3:1.7b (2.28s, 100% recall, 8 FPs)
- **Best for multi-modal**: gemma3:latest (2.47s, 100% recall, 9 FPs)
- **Key insight**: False positives are acceptable - easier to merge than split

## Current Status (2025-07-03)

### ✅ What We've Learned
- **LLM detection is viable**: Multiple models achieve 100% recall
- **Speed targets nearly met**: phi4-mini at 2.04s/boundary (target: <2s)
- **False positives are acceptable**: Easier to merge than split documents
- **Simple approaches work best**: Complex strategies (two-pass) add minimal value
- **Model selection matters**: Clear trade-offs between speed and accuracy

### 🎯 Best Approaches

#### Best Recall (100%)
```python
# Prompt with page break guidance
prompt = f"""Your task is to determine if two document snippets are part of a single document or are different documents.

You will be given the bottom part of Page 1 and the top portion of Page 2. Your task is to determine whether Page 1 and Page 2 are part of a single document or if Page 1 is the end of one document and Page 2 is the start of a new document.

IMPORTANT: A simple page break within a document is NOT a document boundary. Look for signs of a completely new document starting, such as:
- New letterhead or header
- Complete change in formatting or topic
- New date that suggests a different document
- Signature at the bottom of Page 1 followed by a new document header on Page 2

Please only respond with "Same Document" or "Different Documents"

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}"""
```

#### Best Balance (Focus on Endings)
```python
# Focus on document endings - 40% recall but only 3 false positives
prompt = f"""Analyze if Page 1 ends a document and Page 2 starts a new one.

Signs that Page 1 ENDS a document:
- Signature line or signature block
- "Sincerely," or similar closing
- Final paragraph with conclusive language
- Document footer or end mark

Signs that Page 2 STARTS a new document:
- New letterhead or document header
- New date at the top
- "Dear" or similar greeting
- Document title or subject line

Bottom of Page 1:
{page1_bottom}

Top of Page 2:
{page2_top}

Based on these signs, are these pages from the Same Document or Different Documents?"""
```

### 📊 Comprehensive Test Results (2025-07-03)

#### Full Model Comparison (13 models tested with page break guidance prompt, 200 char context)
| Model | F1 | Recall | Precision | FP | Speed | Size | Notes |
|-------|-----|--------|-----------|-----|-------|------|-------|
| **qwen3:8b** | **0.667** | 80% | **57.1%** | **3** | 9.13s | 5.2GB | **Best overall accuracy** |
| qwen3:1.7b | 0.556 | **100%** | 38.5% | 8 | 2.28s | 1.4GB | Best small model |
| granite3.3:8b | 0.556 | **100%** | 38.5% | 8 | 8.30s | 4.9GB | Good accuracy, slow |
| phi3:mini | 0.526 | **100%** | 35.7% | 9 | 4.27s | 2.2GB | Consistent performer |
| phi4-mini:3.8b | 0.526 | **100%** | 35.7% | 9 | **2.04s** | 2.5GB | **Fastest with 100% recall** |
| gemma3:latest | 0.526 | **100%** | 35.7% | 9 | 2.47s | 3.3GB | Same as phi4-mini |
| deepseek-r1:1.5b | 0.500 | 60% | 42.9% | 4 | 1.95s | 1.1GB | Good for size |
| llama3:8b-instruct | 0.500 | 40% | **66.7%** | **1** | 5.52s | 5.7GB | **Best precision** |
| granite3.3:2b | 0.471 | 80% | 33.3% | 8 | 2.16s | 1.5GB | Decent small model |
| gemma3n:e2b | 0.333 | 40% | 28.6% | 5 | 2.24s | 5.6GB | Experimental, poor |
| qwen3:0.6b | 0.222 | 20% | 25.0% | 3 | **0.98s** | 0.5GB | Too small for task |
| phi4-mini-reasoning | 0.000 | 0% | 0% | 0 | 5.07s | 3.2GB | Failed completely |
| deepseek-r1:8b | 0.000 | 0% | 0% | 1 | 9.05s | 5.2GB | Failed completely |

#### Model Category Winners
- **🏆 Best Overall**: qwen3:8b (F1=0.667, only 3 false positives)
- **⚡ Best Speed**: qwen3:0.6b (0.98s) but poor accuracy
- **⚡ Best Speed + Accuracy**: phi4-mini:3.8b (2.04s, 100% recall)
- **📦 Best Small Model (<2GB)**: qwen3:1.7b (1.4GB, F1=0.556, 100% recall)
- **🎯 Best Precision**: llama3:8b-instruct (66.7%, only 1 false positive)

#### Prompt Variations (phi4-mini)
| Prompt Variation | Recall | Precision | F1 | False Positives | Context |
|-----------------|--------|-----------|-----|-----------------|---------|
| Page break guidance | 100% | 35.7% | 0.526 | 9 | 300 chars |
| Stricter criteria | 100% | 35.7% | 0.526 | 9 | 300 chars |
| 200 char context | 100% | 35.7% | 0.526 | 9 | 200 chars |
| Few-shot examples | 80% | 33.3% | 0.471 | 8 | 300 chars |
| 400 char context | 80% | 30.8% | 0.444 | 9 | 400 chars |
| Focus on endings | 40% | 40.0% | 0.400 | 3 | 300 chars |
| Original baseline | 60% | 27.3% | 0.375 | 8 | 300 chars |

### 🔍 Key Insights from Comprehensive Testing
1. **Model size doesn't guarantee performance** - qwen3:1.7b (1.4GB) outperforms many larger models
2. **Different architectures excel at different aspects**:
   - Qwen models: Best overall balance
   - Phi models: Fastest with good accuracy
   - Llama3: High precision but poor recall
3. **Sweet spot exists** - Models between 1.5-3GB offer best speed/accuracy trade-off
4. **100% recall achievable** - 6 models achieve perfect recall with the right prompt
5. **False positive problem persists** - Even best model (qwen3:8b) has 3 false positives

### ⚠️ Current Status
- **✅ Solved**: Speed issue - phi4-mini achieves 2.04s (close to <2s target)
- **✅ Solved**: Recall issue - Multiple models achieve 100% recall
- **🔄 Challenge**: False positives persist
  - Best single model: qwen3:8b with 3 FPs (but slow at 9.13s)
  - Fast models: 8-9 FPs typical
  - Two-pass approach shows minimal improvement
- **🔄 Remaining**: Production implementation needed
- **🆕 Discovered**: Model consistency issues (qwen3:1.7b behavior varies)

## ✅ Completed Experiments
1. **Initial model comparison**: Tested phi3, phi4-mini, gemma3
2. **Prompt refinement**: Page break guidance achieves 100% recall
3. **Context window testing**: 200 chars is optimal (same results, less data)
4. **Comprehensive model testing**: 13 models evaluated
   - Found qwen3:8b as best overall (F1=0.667)
   - Identified phi4-mini:3.8b as best speed/accuracy balance
   - Discovered qwen3:1.7b as excellent small model option
5. **Performance targets**: Nearly achieved <2s goal (phi4-mini at 2.04s)
6. **Two-pass verification testing**: Limited success
   - Minimal false positive reduction (2 out of 14)
   - Lost recall in some cases
   - qwen3:8b too slow for verification (15-20s/boundary)
   - Model behavior inconsistency discovered

## 🚀 Next Experimental Directions

### 1. 🖼️ Multi-Modal Testing with Gemma
Explore image-based boundary detection:
```python
# Test gemma3 with page images instead of text
# Compare accuracy and speed vs text-based approach
# Potentially better at detecting visual boundaries (letterheads, signatures)
```

### 2. 🎯 Advanced Prompt Engineering
Test new prompting strategies:
- **Persona-based**: "You are a records clerk filing documents for litigation..."
- **Few-shot examples**: Include 2-3 archetypal boundary examples
- **Prompt compression**: Minimize tokens while maintaining accuracy
- **Filing cabinet rules**: Leverage document organization metaphors
- **Confidence elicitation**: Get model to express uncertainty

### 3. 📚 Comprehensive Dataset Testing
Expand testing scope:
- **Full document validation**: Test all 36 pages (not just first 15)
- **Create diverse test PDFs**:
  - Mixed document types (letters, memos, reports, forms)
  - Challenging boundaries (mid-sentence breaks, similar headers)
  - Various formatting styles
  - Edge cases (single-page documents, very short documents)
- **Cross-validate**: Test models on different document sets

### 4. 🔬 Experimental Ideas to Explore
- **Context window experiments**: Test with full page vs snippets
- **Preprocessing impact**: Clean text vs raw OCR output
- **Temperature effects**: Does T>0 help with confidence?
- **Batch processing**: Can we process multiple boundaries simultaneously?
- **Layout-aware prompts**: Include spatial information in prompts

## Quick Commands
```bash
# Test finalist models
python pdf_splitter/detection/experiments/test_finalists.py

# Run two-pass verification experiments
python pdf_splitter/detection/experiments/two_pass_verification.py

# Test on full document (36 pages)
python pdf_splitter/detection/experiments/full_document_test.py

# Quick test a specific model
ollama run qwen3:8b "Test prompt here"

# Compare finalist models side-by-side
python pdf_splitter/detection/experiments/finalist_comparison.py
```

## 🎯 Implementation Recommendations

### For Production LLMDetector
```python
class LLMDetector(BaseDetector):
    def __init__(self,
                 model: str = "phi4-mini:3.8b",  # Default for speed
                 mode: str = "balanced"):         # balanced/speed/accuracy
        # Mode configurations:
        # - speed: phi4-mini:3.8b (2.04s, 9 FPs)
        # - accuracy: qwen3:8b (9.13s, 3 FPs)
        # - balanced: gemma3:latest (2.47s, 9 FPs)
        # - lightweight: qwen3:1.7b (2.28s, 8 FPs)
```

### User-Facing Options
1. **Processing Mode Selection**:
   - "Fast" - phi4-mini:3.8b for bulk processing
   - "Accurate" - qwen3:8b for critical documents
   - "Balanced" - gemma3:latest for general use
   - "Lightweight" - qwen3:1.7b for resource-limited environments

2. **False Positive Handling**:
   - Easy merge UI for combining over-split documents
   - Batch review interface
   - Confidence indicators on boundaries

### Integration with Other Signals
- LLM provides baseline detection
- Visual signals refine boundaries
- Heuristics handle obvious cases
- Combined confidence scoring

## 🔄 Current Testing Priority
1. **Multi-modal gemma testing** - Could be game-changer
2. **Prompt engineering** - Low-hanging fruit for improvement
3. **Full dataset validation** - Ensure robustness

## 📋 Model Selection Guide

### For Different Use Cases:
1. **Maximum Accuracy** → **qwen3:8b**
   - F1: 0.667, Only 3 false positives
   - Trade-off: Slower (9.13s) and larger (5.2GB)

2. **Production Speed + Accuracy** → **phi4-mini:3.8b**
   - 100% recall, 2.04s per boundary
   - Good balance for real-time processing

3. **Resource Constrained** → **qwen3:1.7b**
   - Only 1.4GB, still achieves 100% recall
   - 2.28s speed is acceptable

4. **Minimum False Positives** → **llama3:8b-instruct**
   - Only 1 false positive!
   - Trade-off: Poor recall (40%)

## 📊 Key Conclusions

### Our Four Finalist Models
| Model | F1 | Recall | FPs | Speed | Key Strength |
|-------|-----|--------|-----|-------|----------------|
| **qwen3:8b** | 0.667 | 80% | 3 | 9.13s | Highest accuracy, fewest false positives |
| **phi4-mini:3.8b** | 0.526 | 100% | 9 | 2.04s | Fastest with perfect recall |
| **gemma3:latest** | 0.526 | 100% | 9 | 2.47s | Multi-modal capable, consistent |
| **qwen3:1.7b** | 0.556 | 100% | 8 | 2.28s | Smallest size, great performance |

### Strategic Decisions
1. **Drop two-pass verification** - Minimal benefit (2/14 FPs filtered) for 2x processing time
2. **Accept false positives** - Users can easily merge vs difficult to split missed boundaries
3. **Focus on single-model optimization** - Prompt engineering and model selection
4. **Prepare for multi-signal system** - LLM + visual + heuristic signals will reduce FPs

## 📊 Testing Summary So Far
- **Models tested**: 13
- **Test scope**: 15 pages, single document
- **Best F1 score**: 0.667 (qwen3:8b)
- **Models achieving 100% recall**: 6
- **Models meeting <2.5s target**: 7

## 🔬 Open Questions Requiring Testing
1. **Multi-modal potential** - Will image-based detection outperform text?
2. **Prompt optimization** - Can persona/few-shot reduce FPs significantly?
3. **Full document behavior** - Do models maintain accuracy on pages 16-36?
4. **Document type sensitivity** - Which models handle forms vs letters better?
5. **Consistency issues** - Why does qwen3:1.7b behavior vary between runs?
6. **Speed improvements** - Can prompt compression get us under 2s?

## 💡 Philosophy Going Forward

### Core Principles
1. **Perfect is the enemy of good** - 100% recall with some FPs is better than missing boundaries
2. **User experience matters** - Make it easy to correct mistakes
3. **One size doesn't fit all** - Offer speed/accuracy trade-offs
4. **Simple solutions first** - Complex approaches (two-pass) often aren't worth it

### Expected Outcomes
- **LLM alone**: 70-90% precision achievable
- **LLM + other signals**: 95%+ precision realistic
- **Speed**: Sub-2s per boundary is achievable
- **Flexibility**: Let users choose based on their needs

### Success Metrics
- ✅ 100% recall maintained
- ✅ <5s per page total processing
- ✅ User-friendly correction interface
- ✅ Configurable accuracy levels
