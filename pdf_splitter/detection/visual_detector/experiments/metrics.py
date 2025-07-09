"""
Metrics and evaluation utilities for visual boundary detection experiments.

This module provides functions for calculating and visualizing
performance metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_confusion_matrix(
    y_true: List[bool],
    y_pred: List[bool],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot a confusion matrix for boundary detection results.

    Args:
        y_true: True boundary labels
        y_pred: Predicted boundary labels
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["No Boundary", "Boundary"],
        yticklabels=["No Boundary", "Boundary"],
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(
    y_true: List[bool],
    y_scores: List[float],
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
) -> float:
    """
    Plot ROC curve for boundary detection.

    Args:
        y_true: True boundary labels
        y_scores: Similarity scores (or confidence scores)
        title: Title for the plot
        save_path: Optional path to save the plot

    Returns:
        Area under the ROC curve (AUC)
    """
    # For boundary detection, lower similarity means boundary
    # So we need to invert the scores
    y_scores_inverted = [-score for score in y_scores]

    fpr, tpr, _ = roc_curve(y_true, y_scores_inverted)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    plt.show()

    return roc_auc


def calculate_threshold_sweep(
    similarities: List[float],
    true_boundaries: List[bool],
    thresholds: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """
    Calculate metrics across different thresholds.

    Args:
        similarities: Similarity scores between pages
        true_boundaries: True boundary labels
        thresholds: Optional list of thresholds to test

    Returns:
        Dictionary with metrics for each threshold
    """
    if thresholds is None:
        # Use percentiles of similarity scores
        thresholds = np.percentile(similarities, np.arange(10, 100, 10))

    results = {
        "thresholds": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "accuracy": [],
    }

    for threshold in thresholds:
        # Predict boundaries where similarity < threshold
        predictions = [sim < threshold for sim in similarities]

        # Calculate metrics
        tp = sum(1 for t, p in zip(true_boundaries, predictions) if t and p)
        fp = sum(1 for t, p in zip(true_boundaries, predictions) if not t and p)
        tn = sum(1 for t, p in zip(true_boundaries, predictions) if not t and not p)
        fn = sum(1 for t, p in zip(true_boundaries, predictions) if t and not p)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (tp + tn) / len(true_boundaries) if len(true_boundaries) > 0 else 0

        results["thresholds"].append(threshold)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1_score"].append(f1)
        results["accuracy"].append(accuracy)

    return results


def plot_threshold_analysis(
    threshold_results: Dict[str, List[float]],
    metric: str = "f1_score",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot how metrics change with threshold.

    Args:
        threshold_results: Results from calculate_threshold_sweep
        metric: Which metric to highlight
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    # Plot all metrics
    for key in ["precision", "recall", "f1_score", "accuracy"]:
        if key in threshold_results:
            plt.plot(
                threshold_results["thresholds"],
                threshold_results[key],
                label=key.replace("_", " ").title(),
                linewidth=2 if key == metric else 1,
                alpha=1.0 if key == metric else 0.7,
            )

    # Find and mark optimal threshold
    if metric in threshold_results:
        best_idx = np.argmax(threshold_results[metric])
        best_threshold = threshold_results["thresholds"][best_idx]
        best_value = threshold_results[metric][best_idx]

        plt.axvline(x=best_threshold, color="red", linestyle="--", alpha=0.5)
        plt.text(
            best_threshold,
            best_value,
            f"  Best {metric}: {best_value:.3f}\n  Threshold: {best_threshold:.3f}",
            verticalalignment="bottom",
        )

    plt.xlabel("Similarity Threshold")
    plt.ylabel("Score")
    plt.title("Performance Metrics vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def generate_summary_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Generate a summary report of all experiments.

    Args:
        results: List of experiment results
        output_path: Path to save the report
    """
    report = {
        "summary": {
            "total_experiments": len(results),
            "timestamp": datetime.now().isoformat(),
        },
        "experiments": [],
    }

    # Sort by F1 score
    sorted_results = sorted(
        results, key=lambda x: x.get("summary", {}).get("f1_score", 0), reverse=True
    )

    for result in sorted_results:
        exp_summary = {
            "technique": result["technique_name"],
            "parameters": result["parameters"],
            "metrics": result["summary"],
            "timestamp": result["timestamp"],
        }
        report["experiments"].append(exp_summary)

    # Find best overall
    if sorted_results:
        best = sorted_results[0]
        report["best_technique"] = {
            "name": best["technique_name"],
            "f1_score": best["summary"]["f1_score"],
            "parameters": best["parameters"],
        }

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Summary report saved to {output_path}")


def print_detailed_metrics(result: Dict[str, Any]) -> None:
    """
    Print detailed metrics for an experiment result.

    Args:
        result: Experiment result dictionary
    """
    print(f"\n=== {result['technique_name']} ===")
    print(f"Parameters: {result['parameters']}")
    print("\nMetrics:")

    metrics = result["metrics"]
    summary = result.get("summary", {})

    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"\n  Precision: {summary.get('precision', 0):.3f}")
    print(f"  Recall:    {summary.get('recall', 0):.3f}")
    print(f"  F1 Score:  {summary.get('f1_score', 0):.3f}")
    print(f"  Accuracy:  {summary.get('accuracy', 0):.3f}")
    print(f"\n  Avg Time/Page: {summary.get('avg_time_per_page', 0):.3f}s")
    print(f"  Total Time:    {metrics['total_processing_time']:.2f}s")
