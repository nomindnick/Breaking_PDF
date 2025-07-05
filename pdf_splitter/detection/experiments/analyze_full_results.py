#!/usr/bin/env python3
"""Analyze the full test results in detail."""

import json
from pathlib import Path
from collections import defaultdict

# Load the results
results_file = Path("pdf_splitter/detection/experiments/results/optimal_prompt_test_20250704_173553.json")
with open(results_file) as f:
    data = json.load(f)

print("=" * 80)
print("COMPREHENSIVE PERFORMANCE ANALYSIS")
print("=" * 80)

# Analyze by prompt across all models
prompt_performance = defaultdict(lambda: {"models": {}, "avg_f1": 0, "avg_accuracy": 0})

for model, model_data in data["models"].items():
    for category in ["baseline", "optimal", "cod"]:
        for prompt_name, prompt_data in model_data.get(category, {}).items():
            if "overall" in prompt_data:
                overall = prompt_data["overall"]
                prompt_performance[prompt_name]["models"][model] = {
                    "accuracy": overall.get("accuracy", 0),
                    "f1_score": overall.get("f1_score", 0),
                    "precision": overall.get("precision", 0),
                    "recall": overall.get("recall", 0),
                    "total_cases": overall.get("total", 0)
                }

# Calculate averages
for prompt_name, perf_data in prompt_performance.items():
    if perf_data["models"]:
        avg_f1 = sum(m["f1_score"] for m in perf_data["models"].values()) / len(perf_data["models"])
        avg_acc = sum(m["accuracy"] for m in perf_data["models"].values()) / len(perf_data["models"])
        perf_data["avg_f1"] = avg_f1
        perf_data["avg_accuracy"] = avg_acc

# Print results by prompt
print("\n1. PROMPT PERFORMANCE RANKING (by F1 Score)")
print("-" * 80)
print(f"{'Prompt':<30} {'Avg F1':<10} {'Avg Acc':<10} {'Models Tested'}")
print("-" * 80)

sorted_prompts = sorted(prompt_performance.items(), key=lambda x: x[1]["avg_f1"], reverse=True)
for prompt_name, perf_data in sorted_prompts:
    models_tested = len(perf_data["models"])
    print(f"{prompt_name:<30} {perf_data['avg_f1']:<10.3f} {perf_data['avg_accuracy']:<10.2%} {models_tested}")

# Analyze by difficulty level
print("\n2. PERFORMANCE BY DIFFICULTY LEVEL")
print("-" * 80)

difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0})

for model, model_data in data["models"].items():
    for category in ["baseline", "cod"]:
        for prompt_name, prompt_data in model_data.get(category, {}).items():
            if "by_difficulty" in prompt_data:
                for diff_level, diff_data in prompt_data["by_difficulty"].items():
                    if prompt_name != "A1_asymmetric":  # Skip the broken prompt
                        difficulty_stats[int(diff_level)]["total"] += diff_data["total"]
                        difficulty_stats[int(diff_level)]["correct"] += diff_data["correct"]

print(f"{'Difficulty':<15} {'Total Cases':<15} {'Correct':<15} {'Accuracy'}")
print("-" * 80)
for diff in sorted(difficulty_stats.keys()):
    stats = difficulty_stats[diff]
    if stats["total"] > 0:
        accuracy = stats["correct"] / stats["total"]
        print(f"{diff:<15} {stats['total']:<15} {stats['correct']:<15} {accuracy:.2%}")

# Best performing combinations
print("\n3. TOP PERFORMING COMBINATIONS")
print("-" * 80)
print(f"{'Model':<20} {'Prompt':<25} {'F1 Score':<10} {'Accuracy':<10} {'Recall'}")
print("-" * 80)

top_combinations = []
for model, model_data in data["models"].items():
    for category in ["baseline", "cod"]:
        for prompt_name, prompt_data in model_data.get(category, {}).items():
            if "overall" in prompt_data and prompt_data["overall"].get("f1_score", 0) > 0:
                top_combinations.append({
                    "model": model,
                    "prompt": prompt_name,
                    "f1": prompt_data["overall"].get("f1_score", 0),
                    "accuracy": prompt_data["overall"].get("accuracy", 0),
                    "recall": prompt_data["overall"].get("recall", 0)
                })

top_combinations.sort(key=lambda x: x["f1"], reverse=True)
for combo in top_combinations[:10]:
    print(f"{combo['model']:<20} {combo['prompt']:<25} {combo['f1']:<10.3f} {combo['accuracy']:<10.2%} {combo['recall']:.2%}")

# Key findings
print("\n4. KEY FINDINGS")
print("-" * 80)

# Count DIFFERENT vs SAME predictions
prediction_counts = defaultdict(lambda: {"Different": 0, "Same": 0, "Unknown": 0})
for model, model_data in data["models"].items():
    for category in ["baseline", "cod"]:
        for prompt_name, prompt_data in model_data.get(category, {}).items():
            if "by_difficulty" in prompt_data:
                for diff_data in prompt_data["by_difficulty"].values():
                    for detail in diff_data.get("details", []):
                        if "predicted" in detail:
                            prediction_counts[prompt_name][detail["predicted"]] += 1

print("\nPrediction Distribution by Prompt:")
print(f"{'Prompt':<30} {'Different':<12} {'Same':<12} {'Unknown'}")
print("-" * 80)
for prompt_name, counts in prediction_counts.items():
    total = sum(counts.values())
    if total > 0:
        print(f"{prompt_name:<30} {counts['Different']:<12} {counts['Same']:<12} {counts['Unknown']}")

# Issues found
print("\n5. ISSUES IDENTIFIED")
print("-" * 80)
print("- A1_asymmetric prompt returns 'Unknown' for all predictions (parsing issue)")
print("- Optimal prompts (phi4_optimal, gemma3_optimal) were not tested due to naming mismatch")
print("- Overall F1 scores are low, suggesting models struggle with boundary detection")
print("- E1_cod_reasoning shows best F1 score (0.500) but low accuracy (11.54%)")
print("- Models tend to over-predict 'Same' for most prompts")

print("\n6. RECOMMENDATIONS")
print("-" * 80)
print("1. Fix the naming issue to test phi4_optimal and gemma3_optimal prompts")
print("2. Debug A1_asymmetric response parsing")
print("3. The Chain-of-Draft prompts (E1, E2) show promise with better F1 scores")
print("4. Consider adjusting prompts to reduce 'Same' bias")
print("5. Test with more balanced datasets or adjust evaluation metrics")