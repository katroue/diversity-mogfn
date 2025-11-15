#!/usr/bin/env python3
"""
Fix MCE computation for flow_matching experiments.

The original MCE computation used auto-tuned eps which selected unreasonably
large values for flow_matching due to many duplicate solutions. This script
recomputes MCE with a more appropriate eps value.

Usage:
    python scripts/ablations/fix_mce_flow_matching.py --dry-run
    python scripts/ablations/fix_mce_flow_matching.py
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def mode_coverage_entropy_fixed(objectives, eps=0.05, min_samples=5):
    """
    Compute MCE with fixed eps to avoid auto-tuning issues.

    Args:
        objectives: Array of shape (N, num_objectives)
        eps: DBSCAN radius (default: 0.05, reasonable for normalized objectives)
        min_samples: Minimum samples per cluster

    Returns:
        mce: Mode coverage entropy
        num_modes: Number of discovered modes
    """
    from sklearn.cluster import DBSCAN

    N = len(objectives)

    if N < min_samples:
        return 0.0, 0

    # Adjust min_samples if dataset is too small
    effective_min_samples = min(min_samples, max(2, N // 5))

    # Cluster with DBSCAN using fixed eps
    clustering = DBSCAN(eps=eps, min_samples=effective_min_samples)
    labels = clustering.fit_predict(objectives)

    # Get cluster distribution (excluding noise label -1)
    unique_labels = set(labels) - {-1}
    num_modes = len(unique_labels)

    if num_modes <= 1:
        return 0.0, num_modes

    # Compute entropy
    cluster_sizes = []
    for label in unique_labels:
        cluster_sizes.append(np.sum(labels == label))

    probs = np.array(cluster_sizes) / N
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize by maximum entropy
    max_entropy = np.log2(num_modes)
    mce = entropy / max_entropy if max_entropy > 0 else 0.0

    return mce, num_modes


def fix_mce_for_experiment(exp_dir: Path, eps: float = 0.05, dry_run: bool = False):
    """
    Fix MCE for a single experiment.

    Args:
        exp_dir: Experiment directory
        eps: DBSCAN eps parameter
        dry_run: If True, don't save changes

    Returns:
        Dictionary with results
    """
    objectives_file = exp_dir / 'objectives.npy'
    metrics_file = exp_dir / 'metrics.json'

    if not objectives_file.exists() or not metrics_file.exists():
        return {
            'exp_name': exp_dir.name,
            'status': 'missing_files',
            'old_mce': None,
            'new_mce': None
        }

    try:
        # Load objectives and metrics
        objectives = np.load(objectives_file)
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        old_mce = metrics.get('mce', None)
        old_num_modes = metrics.get('num_modes', None)

        # Recompute MCE with fixed eps
        new_mce, new_num_modes = mode_coverage_entropy_fixed(objectives, eps=eps)

        # Update metrics if not dry run
        if not dry_run:
            metrics['mce'] = float(new_mce)
            metrics['num_modes'] = int(new_num_modes)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        return {
            'exp_name': exp_dir.name,
            'status': 'success',
            'old_mce': old_mce,
            'old_num_modes': old_num_modes,
            'new_mce': float(new_mce),
            'new_num_modes': int(new_num_modes),
            'change': new_mce - old_mce if old_mce is not None else None
        }

    except Exception as e:
        return {
            'exp_name': exp_dir.name,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Fix MCE computation for flow_matching experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--eps',
        type=float,
        default=0.05,
        help='DBSCAN eps parameter (default: 0.05)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without saving'
    )

    args = parser.parse_args()

    base_dir = Path('results/ablations/loss/base_loss_comparison')

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    print("="*80)
    print("FIXING MCE FOR FLOW_MATCHING EXPERIMENTS")
    print("="*80)
    print(f"Directory: {base_dir}")
    print(f"DBSCAN eps: {args.eps}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Find flow_matching experiments
    fm_dirs = sorted([d for d in base_dir.iterdir()
                     if d.is_dir() and 'flow_matching' in d.name])

    print(f"Found {len(fm_dirs)} flow_matching experiments\n")

    # Process each experiment
    results = []
    for exp_dir in fm_dirs:
        result = fix_mce_for_experiment(exp_dir, eps=args.eps, dry_run=args.dry_run)
        results.append(result)

        if result['status'] == 'success':
            print(f"{'[DRY RUN] ' if args.dry_run else ''}✓ {result['exp_name']}")
            print(f"    Old: MCE={result['old_mce']:.4f}, modes={result['old_num_modes']}")
            print(f"    New: MCE={result['new_mce']:.4f}, modes={result['new_num_modes']}")
            if result['change'] is not None:
                print(f"    Change: {result['change']:+.4f}")
        else:
            print(f"  ✗ {result['exp_name']}: {result['status']}")

    print()
    print("-"*80)
    print("SUMMARY")
    print("-"*80)

    success = [r for r in results if r['status'] == 'success']
    print(f"Fixed: {len(success)}/{len(fm_dirs)} experiments")

    if success:
        changes = [r['change'] for r in success if r['change'] is not None]
        if changes:
            print(f"\nMCE changes:")
            print(f"  Mean: {np.mean(changes):+.4f}")
            print(f"  Range: [{np.min(changes):+.4f}, {np.max(changes):+.4f}]")

    # Regenerate results.csv if changes made
    if not args.dry_run and success:
        print(f"\nRegenerating results.csv...")

        # Load all metrics
        metrics_files = sorted(base_dir.glob('*/metrics.json'))
        all_metrics = []
        for mf in metrics_files:
            try:
                with open(mf, 'r') as f:
                    all_metrics.append(json.load(f))
            except:
                pass

        df = pd.DataFrame(all_metrics)
        if 'exp_name' in df.columns:
            df = df.sort_values('exp_name')

        results_csv = base_dir / 'results.csv'
        df.to_csv(results_csv, index=False)
        print(f"✓ Saved: {results_csv}")

    print(f"\n{'✓ Done! (DRY RUN)' if args.dry_run else '✓ Done!'}")


if __name__ == '__main__':
    main()
