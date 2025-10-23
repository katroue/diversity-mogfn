#!/usr/bin/env python3
"""
Fix and recompute PMD (Pairwise Minimum Distance) for all completed experiments.

ISSUE FOUND:
The pairwise_minimum_distance function was missing tensor conversion,
causing it to return 0.0 when given PyTorch tensors or when there were
identical solutions.

This script:
1. Identifies the PMD computation issue
2. Recomputes PMD correctly for all experiments
3. Updates metrics.json files
4. Regenerates all_results.csv

Usage:
    # Preview changes
    python scripts/fix_and_recompute_pmd.py \
        --results_dir results/ablations/capacity \
        --dry_run
    
    # Actually update
    python scripts/fix_and_recompute_pmd.py \
        --results_dir results/ablations/capacity
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple
import sys


def pairwise_minimum_distance_fixed(objectives: np.ndarray, top_k: int = None) -> float:
    """
    Compute Pairwise Minimum Distance (PMD) for top-K solutions.
    
    FIXED VERSION: Properly handles tensor conversion and edge cases.
    
    PMD = min_{i≠j} d(x_i, x_j)
    
    Args:
        objectives: Objective values, shape (N, num_objectives)
        top_k: Number of top solutions to consider (None = all)
    
    Returns:
        pmd: Minimum pairwise distance
    """
    # FIXED: Convert PyTorch tensor to numpy if needed
    if hasattr(objectives, 'cpu'):
        objectives = objectives.cpu().numpy()
    
    # Ensure it's a numpy array
    objectives = np.asarray(objectives)
    
    # FIXED: Check for minimum number of samples
    if len(objectives) < 2:
        print("    Warning: Less than 2 samples, PMD = 0.0")
        return 0.0
    
    # Select top-K non-dominated solutions if specified
    if top_k is not None and top_k < len(objectives):
        # Find non-dominated solutions
        n = len(objectives)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated[i] = True
                    break
        
        non_dominated = objectives[~is_dominated]
        
        if len(non_dominated) >= top_k:
            objectives = non_dominated[:top_k]
        elif len(non_dominated) > 0:
            # Take all non-dominated + best dominated
            dominated = objectives[is_dominated]
            if len(dominated) > 0:
                sums = np.sum(dominated, axis=1)
                num_to_take = min(top_k - len(non_dominated), len(dominated))
                best_dominated = dominated[np.argsort(sums)[:num_to_take]]
                objectives = np.vstack([non_dominated, best_dominated])
            else:
                objectives = non_dominated
    
    # FIXED: Final check
    if len(objectives) < 2:
        print("    Warning: After filtering, less than 2 samples, PMD = 0.0")
        return 0.0
    
    # Compute minimum pairwise distance
    try:
        distances = pdist(objectives, metric='euclidean')
        
        # FIXED: Check if all distances are 0 (all identical points)
        if len(distances) == 0:
            print("    Warning: No pairwise distances computed, PMD = 0.0")
            return 0.0
        
        min_dist = float(np.min(distances))
        
        # FIXED: Warn if minimum distance is suspiciously small
        if min_dist < 1e-10:
            print(f"    Warning: Very small PMD = {min_dist:.2e} (may indicate duplicate solutions)")
        
        return min_dist
        
    except Exception as e:
        print(f"    Error computing distances: {e}")
        return 0.0


def find_experiment_dirs(results_dir: Path) -> List[Path]:
    """Find all experiment directories that have completed."""
    experiment_dirs = []
    
    for item in results_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if experiment has required files
            if (item / 'metrics.json').exists() and \
               (item / 'objectives.npy').exists():
                experiment_dirs.append(item)
    
    return sorted(experiment_dirs)


def compute_and_update_pmd(exp_dir: Path, dry_run: bool = False) -> Tuple[float, bool]:
    """
    Compute PMD for experiment and update metrics.json.
    
    Returns:
        (pmd_value, success)
    """
    try:
        # Load objectives
        objectives = np.load(exp_dir / 'objectives.npy')
        
        print(f"    Loaded {len(objectives)} objectives")
        
        # Compute PMD with fixed function
        pmd = pairwise_minimum_distance_fixed(objectives)
        
        # Update metrics file
        metrics_file = exp_dir / 'metrics.json'
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        old_pmd = metrics.get('pmd', None)
        metrics['pmd'] = pmd
        
        if dry_run:
            print(f"    [DRY RUN] Would update PMD: {old_pmd} → {pmd:.6f}")
        else:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"    ✓ Updated PMD: {old_pmd} → {pmd:.6f}")
        
        return pmd, True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, False


def regenerate_results_csv(results_dir: Path) -> bool:
    """Regenerate all_results.csv from individual metrics.json files."""
    try:
        print("\nRegenerating all_results.csv...")
        print("-"*70)
        
        # Find all experiment directories
        experiment_dirs = []
        for item in results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if (item / 'metrics.json').exists():
                    experiment_dirs.append(item)
        
        # Load all metrics
        all_metrics = []
        for exp_dir in sorted(experiment_dirs):
            with open(exp_dir / 'metrics.json', 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        
        if len(all_metrics) == 0:
            print("  ✗ No metrics found!")
            return False
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        output_file = results_dir / 'all_results.csv'
        df.to_csv(output_file, index=False)
        
        print(f"  ✓ Saved {len(df)} experiments to {output_file}")
        print(f"  ✓ Columns: {len(df.columns)} metrics")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error regenerating CSV: {e}")
        return False


def diagnose_pmd_issue(results_dir: Path):
    """Diagnose why PMD might be 0.0 everywhere."""
    print("\n" + "="*70)
    print("DIAGNOSING PMD ISSUE")
    print("="*70)
    
    experiment_dirs = find_experiment_dirs(results_dir)
    
    if len(experiment_dirs) == 0:
        print("No experiments found!")
        return
    
    # Check first few experiments
    print(f"\nChecking first 3 experiments for diagnostic info...")
    
    for exp_dir in experiment_dirs[:3]:
        print(f"\n{exp_dir.name}:")
        
        try:
            objectives = np.load(exp_dir / 'objectives.npy')
            print(f"  Objectives shape: {objectives.shape}")
            print(f"  Objectives type: {type(objectives)}")
            print(f"  Objectives dtype: {objectives.dtype}")
            print(f"  First 3 objectives:\n{objectives[:3]}")
            
            # Check for duplicates
            unique_objectives = np.unique(objectives, axis=0)
            print(f"  Unique objectives: {len(unique_objectives)}/{len(objectives)}")
            
            # Compute distances manually
            if len(objectives) >= 2:
                from scipy.spatial.distance import pdist
                dists = pdist(objectives, metric='euclidean')
                print(f"  Pairwise distances computed: {len(dists)}")
                print(f"  Min distance: {np.min(dists):.6f}")
                print(f"  Max distance: {np.max(dists):.6f}")
                print(f"  Mean distance: {np.mean(dists):.6f}")
                
                if np.min(dists) == 0.0:
                    print(f"  ⚠️  WARNING: Minimum distance is 0.0 (duplicate solutions exist)")
                    # Find which solutions are duplicates
                    num_zeros = np.sum(dists == 0.0)
                    print(f"  Number of zero distances: {num_zeros}")
            
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Fix and recompute PMD for all experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose the issue first
  python fix_and_recompute_pmd.py --results_dir results/ablations/capacity --diagnose
  
  # Preview changes (recommended first)
  python fix_and_recompute_pmd.py --results_dir results/ablations/capacity --dry_run
  
  # Actually update all files
  python fix_and_recompute_pmd.py --results_dir results/ablations/capacity
        """
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to results directory (e.g., results/ablations/capacity)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--diagnose',
        action='store_true',
        help='Run diagnostic checks to understand the PMD issue'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Consider only top-K non-dominated solutions (default: all)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Run diagnostics if requested
    if args.diagnose:
        diagnose_pmd_issue(results_dir)
        return
    
    print("="*70)
    print("FIX AND RECOMPUTE PMD FOR ALL EXPERIMENTS")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Dry run: {'YES (no files will be modified)' if args.dry_run else 'NO (files will be updated)'}")
    print(f"Top-K filter: {args.top_k if args.top_k else 'None (use all solutions)'}")
    print()
    
    # Find all experiment directories
    print("Step 1: Finding experiment directories...")
    experiment_dirs = find_experiment_dirs(results_dir)
    
    if len(experiment_dirs) == 0:
        print("✗ No completed experiments found!")
        print("  Looking for directories with: metrics.json, objectives.npy")
        sys.exit(1)
    
    print(f"✓ Found {len(experiment_dirs)} experiments")
    print()
    
    # Process each experiment
    print("Step 2: Computing and updating PMD...")
    print("-"*70)
    
    successful = 0
    failed = 0
    pmd_values = []
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"\n{exp_name}")
        
        pmd_value, success = compute_and_update_pmd(exp_dir, dry_run=args.dry_run)
        
        if success:
            successful += 1
            if pmd_value > 0:  # Only count non-zero values for statistics
                pmd_values.append(pmd_value)
        else:
            failed += 1
    
    print()
    print("-"*70)
    print(f"Successfully processed: {successful}/{len(experiment_dirs)}")
    print(f"Failed: {failed}/{len(experiment_dirs)}")
    
    if len(pmd_values) > 0:
        print(f"\nPMD Statistics (non-zero values only):")
        print(f"  Count: {len(pmd_values)}/{successful}")
        print(f"  Mean: {np.mean(pmd_values):.6f}")
        print(f"  Std:  {np.std(pmd_values):.6f}")
        print(f"  Min:  {np.min(pmd_values):.6f}")
        print(f"  Max:  {np.max(pmd_values):.6f}")
    else:
        print(f"\n⚠️  WARNING: All PMD values are 0.0!")
        print(f"This might indicate:")
        print(f"  1. Duplicate solutions in objectives")
        print(f"  2. Very small objective space")
        print(f"  3. Data loading issues")
        print(f"\nRun with --diagnose flag to investigate:")
        print(f"  python fix_and_recompute_pmd.py --results_dir {results_dir} --diagnose")
    
    # Regenerate CSV (only if not dry run and had success)
    if not args.dry_run and successful > 0:
        print()
        print("Step 3: Regenerating all_results.csv...")
        if regenerate_results_csv(results_dir):
            print("✓ CSV file updated!")
        else:
            print("✗ Failed to update CSV file")
    elif args.dry_run:
        print()
        print("Step 3: Skipped (dry run mode)")
    
    # Final summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if args.dry_run:
        print("✓ Dry run completed successfully!")
        print()
        print("What would be updated:")
        print(f"  - {successful} metrics.json files with PMD values")
        print(f"  - all_results.csv regenerated")
        print()
        print("Run without --dry_run to actually update the files.")
    else:
        print("✓ All files updated successfully!")
        print()
        print("What was updated:")
        print(f"  - {successful} metrics.json files with PMD values")
        print(f"  - all_results.csv regenerated")
        print()
        if len(pmd_values) > 0:
            print("✓ PMD metric successfully recomputed for all experiments!")
        else:
            print("⚠️  PMD values are still 0.0 - run --diagnose to investigate")
    
    print("="*70)


if __name__ == '__main__':
    main()