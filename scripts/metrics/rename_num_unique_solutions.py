"""
Rename num_unique_solutions to num_modes in all metrics.json files.

This script updates the field name from the old 'num_unique_solutions'
to the standardized 'num_modes' across all experiment results.
"""

import json
from pathlib import Path
from typing import Dict, Any


def rename_field_in_metrics(metrics_path: Path) -> bool:
    """
    Rename num_unique_solutions to num_modes in a metrics.json file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Check if num_unique_solutions exists
        if 'num_unique_solutions' not in metrics:
            return False

        # Rename the field
        metrics['num_modes'] = metrics.pop('num_unique_solutions')

        # Save back
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return True

    except Exception as e:
        print(f"  ERROR processing {metrics_path}: {e}")
        return False


def main():
    """Find and rename all metrics.json files."""

    print("=" * 80)
    print("RENAMING num_unique_solutions → num_modes")
    print("=" * 80)

    # Find all metrics.json files
    results_dir = Path(__file__).parent.parent / "results"
    metrics_files = list(results_dir.rglob("metrics.json"))

    print(f"\nFound {len(metrics_files)} metrics.json files")

    # Process each file
    modified_count = 0
    skipped_count = 0

    for metrics_path in metrics_files:
        was_modified = rename_field_in_metrics(metrics_path)

        if was_modified:
            modified_count += 1
            # Print relative path
            rel_path = metrics_path.relative_to(results_dir)
            print(f"  ✓ {rel_path}")
        else:
            skipped_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files:     {len(metrics_files)}")
    print(f"Modified:        {modified_count}")
    print(f"Skipped:         {skipped_count}")
    print(f"Success rate:    {modified_count / len(metrics_files) * 100:.1f}%")

    if modified_count > 0:
        print("\n✅ Renaming complete!")
        print("\nNext steps:")
        print("1. Update scripts/validation/compute_metric_correlations.py:")
        print("   Change 'num_unique_solutions' → 'num_modes' in METRIC_CATEGORIES")
        print("2. Re-run correlation analysis to verify all metrics are found")


if __name__ == "__main__":
    main()