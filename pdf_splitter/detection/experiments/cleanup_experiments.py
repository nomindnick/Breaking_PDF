#!/usr/bin/env python3
"""
Clean up experimental scripts that are no longer needed.

Keep only the essential utilities and results.
"""

from pathlib import Path


def cleanup_experiments():
    """Remove temporary test scripts, keeping only essential files."""
    experiments_dir = Path(__file__).parent

    # Files to keep
    keep_files = {
        # Core experiment infrastructure
        "experiment_runner.py",
        "run_experiments.py",
        "__init__.py",
        # Essential test utilities
        "working_simple_approach.py",
        "model_performance_comparison.py",
        "comprehensive_model_test.py",
        "prompt_refinement_test.py",
        "two_pass_verification.py",
        # Documentation
        "README.md",
        "EXPERIMENT_SUMMARY.md",
        # This cleanup script
        "cleanup_experiments.py",
    }

    # Directories to keep (currently unused but may be needed later)
    # keep_dirs = {"results", "configs", "prompts", "__pycache__"}

    # Count files
    removed_count = 0
    kept_count = 0

    print("Cleaning up experiments directory...")
    print("-" * 50)

    # List all Python files
    for file_path in experiments_dir.glob("*.py"):
        if file_path.name not in keep_files:
            print(f"Removing: {file_path.name}")
            file_path.unlink()
            removed_count += 1
        else:
            kept_count += 1

    print("\nSummary:")
    print(f"  Files removed: {removed_count}")
    print(f"  Files kept: {kept_count}")

    # Show what's in results directory
    results_dir = experiments_dir / "results"
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json"))
        print(f"\nResults directory contains {len(result_files)} files")

        # Get latest results
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"Latest result: {latest.name}")


if __name__ == "__main__":
    cleanup_experiments()
