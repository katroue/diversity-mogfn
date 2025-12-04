#!/usr/bin/env python3
"""
Recalculate Pareto Front Smoothness (PFS) for all experiments using extended implementation.

The extended PFS implementation now supports:
- 2 objectives: Original 1D curve fitting
- 3+ objectives: PCA projection to 2D manifold, then curve fitting

This script:
1. Finds all experiments with saved objectives
2. Recalculates PFS using the extended implementation
3. Updates metrics.json files with new PFS values
4. Creates a backup of original metrics before updating

Usage:
    # Recalculate PFS for all ablations
    python scripts/metrics/recalculate_pfs.py --results_dir results/factorials

    # Recalculate PFS for specific ablation
    python scripts/metrics/recalculate_pfs.py --results_dir results/ablations/capacity

    # Dry run (show what would be updated)
    python scripts/metrics/recalculate_pfs.py --results_dir results/ablations --dry_run
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.objective import pareto_front_smoothness


def find_experiments(results_dir: Path) -> list:
    """
    Find all experiment directories with objectives.npy files.

    Args:
        results_dir: Root directory to search

    Returns:
        exp_dirs: List of experiment directories
    """
    exp_dirs = []

    for obj_file in results_dir.rglob("objectives.npy"):
        exp_dir = obj_file.parent
        metrics_file = exp_dir / "metrics.json"

        if metrics_file.exists():
            exp_dirs.append(exp_dir)

    return sorted(exp_dirs)


def recalculate_pfs_for_experiment(exp_dir: Path, dry_run: bool = False) -> dict:
    """
    Recalculate PFS for a single experiment.

    Args:
        exp_dir: Experiment directory
        dry_run: If True, don't save changes

    Returns:
        result: Dictionary with old_pfs, new_pfs, and status
    """
    objectives_file = exp_dir / "objectives.npy"
    metrics_file = exp_dir / "metrics.json"

    # Load objectives
    try:
        objectives = np.load(objectives_file)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to load objectives: {e}"
        }

    # Load metrics
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to load metrics: {e}"
        }

    # Get old PFS value
    old_pfs = metrics.get('pfs', None)

    # Calculate new PFS
    try:
        new_pfs = pareto_front_smoothness(objectives)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to calculate PFS: {e}",
            "old_pfs": old_pfs
        }

    # Check if changed
    if old_pfs is not None:
        diff = abs(new_pfs - old_pfs)
        changed = diff > 1e-6
    else:
        changed = True
        diff = None

    result = {
        "status": "success",
        "old_pfs": old_pfs,
        "new_pfs": new_pfs,
        "changed": changed,
        "diff": diff,
        "num_objectives": objectives.shape[1],
        "num_samples": objectives.shape[0]
    }

    # Update metrics if not dry run
    if not dry_run and changed:
        try:
            # Backup original metrics
            backup_file = exp_dir / "metrics_backup.json"
            if not backup_file.exists():
                shutil.copy(metrics_file, backup_file)

            # Update PFS value
            metrics['pfs'] = float(new_pfs)

            # Add metadata about update
            if 'pfs_metadata' not in metrics:
                metrics['pfs_metadata'] = {}

            metrics['pfs_metadata']['recalculated'] = True
            metrics['pfs_metadata']['recalculation_date'] = datetime.now().isoformat()
            metrics['pfs_metadata']['old_value'] = float(old_pfs) if old_pfs is not None else None
            metrics['pfs_metadata']['method'] = 'extended_pfs_with_manifold_projection'

            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            result["saved"] = True
        except (PermissionError, OSError) as e:
            result["saved"] = False
            result["status"] = "error"
            result["error"] = f"Permission denied: {e}"
            return result
    else:
        result["saved"] = False

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate PFS for all experiments using extended implementation"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root directory containing experiment results"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be updated without saving"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        return 1

    print("=" * 80)
    print("RECALCULATING PARETO FRONT SMOOTHNESS (PFS)")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Find all experiments
    print("Finding experiments...")
    exp_dirs = find_experiments(results_dir)
    print(f"Found {len(exp_dirs)} experiments with objectives")
    print()

    if len(exp_dirs) == 0:
        print("❌ No experiments found!")
        return 1

    # Process each experiment
    print("Recalculating PFS...")
    print("-" * 80)

    results = []
    num_updated = 0
    num_unchanged = 0
    num_errors = 0

    for i, exp_dir in enumerate(exp_dirs, 1):
        rel_path = exp_dir.relative_to(results_dir)

        result = recalculate_pfs_for_experiment(exp_dir, dry_run=args.dry_run)
        results.append((rel_path, result))

        if result["status"] == "error":
            num_errors += 1
            print(f"[{i}/{len(exp_dirs)}] ❌ {rel_path}")
            print(f"           Error: {result['error']}")

        elif result["changed"]:
            num_updated += 1
            old = result["old_pfs"]
            new = result["new_pfs"]
            diff = result.get("diff", 0)

            saved_str = "✓ SAVED" if result.get("saved") else "(DRY RUN)"

            old_str = f"{old:.6f}" if old is not None else "None"
            diff_str = f"{diff:.6f}" if diff is not None else "N/A"

            print(f"[{i}/{len(exp_dirs)}] ✓ {rel_path}")
            print(f"           Old PFS: {old_str}")
            print(f"           New PFS: {new:.6f}")
            print(f"           Diff: {diff_str} {saved_str}")

        else:
            num_unchanged += 1
            if i % 10 == 0:  # Only print every 10th unchanged
                print(f"[{i}/{len(exp_dirs)}] - {rel_path} (unchanged)")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total experiments:  {len(exp_dirs)}")
    print(f"Updated:            {num_updated}")
    print(f"Unchanged:          {num_unchanged}")
    print(f"Errors:             {num_errors}")
    print(f"Success rate:       {(num_updated + num_unchanged) / len(exp_dirs) * 100:.1f}%")
    print()

    if args.dry_run:
        print("🔍 DRY RUN - No files were modified")
        print("   Remove --dry_run flag to save changes")
    elif num_updated > 0:
        print("✅ PFS recalculation complete!")
        print(f"   Updated {num_updated} experiments")
        print(f"   Backups saved to metrics_backup.json")
    else:
        print("✅ All PFS values already up to date!")

    print()
    return 0 if num_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())