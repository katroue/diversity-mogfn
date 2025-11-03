#!/usr/bin/env python3
"""
Test script for capacity × sampling factorial experiment.

Quick test with reduced iterations (1000) and seeds (2) to identify
which configurations suffer from mode collapse.

Usage:
    # Run all conditions
    python tests/factorials/test_capacity_sampling_factorial.py

    # Run specific conditions only
    python tests/factorials/test_capacity_sampling_factorial.py \
        --conditions small_low,medium_low

    # Analyze existing results without re-running
    python tests/factorials/test_capacity_sampling_factorial.py --analyze-only
"""

import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_factorial_experiment(config_path: Path,
                            output_dir: Path,
                            conditions: str = None,
                            device: str = 'cpu') -> int:
    """
    Run factorial experiment using the main script.

    Returns:
        Exit code from subprocess
    """
    script_path = project_root / 'scripts' / 'factorials' / 'hypergrid' / 'run_factorial_experiment.py'

    cmd = [
        sys.executable,
        str(script_path),
        '--config', str(config_path),
        '--output_dir', str(output_dir),
        '--device', device,
        '--resume'  # Always use resume to skip completed
    ]

    if conditions:
        cmd.extend(['--conditions', conditions])

    print(f"\n{'='*80}")
    print("RUNNING TEST FACTORIAL EXPERIMENT")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def analyze_mode_collapse(results_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze results to detect mode collapse in each experiment.

    Returns:
        df: DataFrame with collapse statistics
        summary: Dictionary with overall statistics
    """
    print(f"\n{'='*80}")
    print("ANALYZING MODE COLLAPSE")
    print(f"{'='*80}\n")

    results = []

    # Analyze each experiment
    for exp_dir in sorted(results_dir.glob('*_seed*')):
        exp_name = exp_dir.name

        # Load objectives
        obj_file = exp_dir / 'objectives.npy'
        if not obj_file.exists():
            print(f"⚠️  {exp_name}: No objectives.npy found (experiment incomplete)")
            continue

        objectives = np.load(obj_file)

        # Load metrics
        metrics_file = exp_dir / 'metrics.json'
        if not metrics_file.exists():
            print(f"⚠️  {exp_name}: No metrics.json found")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Analyze diversity
        n_unique = len(np.unique(objectives, axis=0))
        n_total = len(objectives)
        unique_rate = n_unique / n_total * 100

        # Detect collapse (1 unique = complete collapse, <10 unique = severe collapse)
        collapsed = n_unique == 1
        severe_collapse = n_unique < 10

        # Extract condition name
        condition = '_'.join(exp_name.split('_')[:-1])
        seed = int(exp_name.split('_seed')[-1])

        # Collect results
        result = {
            'exp_name': exp_name,
            'condition': condition,
            'seed': seed,
            'n_objectives': n_total,
            'n_unique': n_unique,
            'unique_rate': unique_rate,
            'collapsed': collapsed,
            'severe_collapse': severe_collapse,
            'mce': metrics.get('mce', 0.0),
            'pas': metrics.get('pas', 0.0),
            'pmd': metrics.get('pmd', 0.0),
            'hypervolume': metrics.get('hypervolume', 0.0),
            'final_loss': metrics.get('final_loss', float('nan')),
        }

        results.append(result)

        # Print status
        if collapsed:
            print(f"❌ {exp_name}: COLLAPSED (1 unique objective)")
        elif severe_collapse:
            print(f"⚠️  {exp_name}: SEVERE COLLAPSE ({n_unique} unique, {unique_rate:.1f}%)")
        else:
            print(f"✅ {exp_name}: OK ({n_unique} unique, {unique_rate:.1f}%)")

    if not results:
        print("\n⚠️  No completed experiments found!")
        return pd.DataFrame(), {}

    df = pd.DataFrame(results)

    # Compute summary statistics
    summary = {
        'total_experiments': len(df),
        'collapsed_count': df['collapsed'].sum(),
        'collapsed_rate': df['collapsed'].mean() * 100,
        'severe_collapse_count': df['severe_collapse'].sum(),
        'severe_collapse_rate': df['severe_collapse'].mean() * 100,
    }

    return df, summary


def print_summary_by_condition(df: pd.DataFrame) -> None:
    """Print collapse statistics grouped by condition."""
    print(f"\n{'='*80}")
    print("MODE COLLAPSE BY CONDITION")
    print(f"{'='*80}\n")

    if df.empty:
        print("No data to analyze.")
        return

    # Group by condition
    grouped = df.groupby('condition').agg({
        'collapsed': ['sum', 'count', 'mean'],
        'severe_collapse': 'sum',
        'n_unique': ['mean', 'std', 'min', 'max'],
        'mce': ['mean', 'std'],
        'hypervolume': ['mean', 'std'],
    }).round(4)

    # Print condition-wise summary
    for condition in sorted(df['condition'].unique()):
        cond_df = df[df['condition'] == condition]
        n_collapsed = cond_df['collapsed'].sum()
        n_severe = cond_df['severe_collapse'].sum()
        n_total = len(cond_df)

        collapse_rate = n_collapsed / n_total * 100
        severe_rate = n_severe / n_total * 100

        avg_unique = cond_df['n_unique'].mean()
        avg_mce = cond_df['mce'].mean()

        # Determine status
        if collapse_rate > 50:
            status = "❌ FAILED"
        elif collapse_rate > 0 or severe_rate > 50:
            status = "⚠️  PARTIAL"
        else:
            status = "✅ OK"

        print(f"{condition:20s} {status}")
        print(f"  Collapsed: {n_collapsed}/{n_total} ({collapse_rate:5.1f}%)")
        print(f"  Severe:    {n_severe}/{n_total} ({severe_rate:5.1f}%)")
        print(f"  Avg unique objectives: {avg_unique:.1f}")
        print(f"  Avg MCE: {avg_mce:.4f}")
        print()


def print_overall_summary(summary: Dict) -> None:
    """Print overall summary statistics."""
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Complete collapse: {summary['collapsed_count']} ({summary['collapsed_rate']:.1f}%)")
    print(f"Severe collapse:   {summary['severe_collapse_count']} ({summary['severe_collapse_rate']:.1f}%)")
    print()

    # Interpretation
    if summary['collapsed_rate'] > 50:
        print("⚠️  CRITICAL: >50% of experiments have complete mode collapse!")
        print("   This suggests a fundamental training or implementation issue.")
    elif summary['collapsed_rate'] > 20:
        print("⚠️  WARNING: >20% of experiments have complete mode collapse.")
        print("   Some configurations may be inadequate for learning diversity.")
    elif summary['collapsed_rate'] > 0:
        print("ℹ️  INFO: Some experiments collapsed, but most are working.")
        print("   This is expected - factorial design reveals inadequate configs.")
    else:
        print("✅ SUCCESS: No complete mode collapse detected!")
        print("   All configurations are learning diverse solutions.")


def save_analysis(df: pd.DataFrame, summary: Dict, output_dir: Path) -> None:
    """Save analysis results to files."""
    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    # Save detailed results
    if not df.empty:
        csv_file = analysis_dir / 'collapse_analysis.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n📊 Saved detailed analysis to: {csv_file}")

    # Save summary
    summary_file = analysis_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Saved summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Test capacity × sampling factorial for mode collapse',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('tests/factorials/test_capacity_sampling_config.yaml'),
        help='Path to test configuration (default: tests/factorials/test_capacity_sampling_config.yaml)'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('tests/factorials/results'),
        help='Output directory for test results (default: tests/factorials/results)'
    )

    parser.add_argument(
        '--conditions',
        type=str,
        default=None,
        help='Comma-separated list of conditions to test (default: all)'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing results, do not run experiments'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for training (cpu or cuda)'
    )

    args = parser.parse_args()

    # Resolve paths
    config_path = project_root / args.config
    output_dir = project_root / args.output_dir

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    # Run experiments (unless analyze-only)
    if not args.analyze_only:
        exit_code = run_factorial_experiment(
            config_path=config_path,
            output_dir=output_dir,
            conditions=args.conditions,
            device=args.device
        )

        if exit_code != 0:
            print(f"\n❌ Experiment failed with exit code {exit_code}")
            return exit_code
    else:
        print("\n📊 Analyzing existing results (--analyze-only mode)")

    # Analyze results
    df, summary = analyze_mode_collapse(output_dir)

    if not df.empty:
        # Print summaries
        print_summary_by_condition(df)
        print_overall_summary(summary)

        # Save analysis
        save_analysis(df, summary, output_dir)

        print(f"\n{'='*80}")
        print("TEST COMPLETE")
        print(f"{'='*80}\n")

        # Exit with error if >50% collapsed
        if summary['collapsed_rate'] > 50:
            print("❌ Test FAILED: >50% mode collapse rate")
            return 1
    else:
        print("\n⚠️  No results to analyze. Run without --analyze-only first.")
        return 1

    print("✅ Test PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
