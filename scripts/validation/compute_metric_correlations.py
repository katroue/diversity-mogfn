#!/usr/bin/env python3
"""
Compute correlation matrices for diversity metrics.

This script implements Validation 1.2: Correlation Matrix Analysis from
Phase 4: Metric Validation.

It computes correlation matrices across all diversity metrics to identify:
- Redundant metrics (|r| > 0.9)
- Independent metrics (|r| < 0.3)
- Metric relationships and dependencies

Usage:
    # Analyze ablation studies
    python scripts/validation/compute_metric_correlations.py --dataset ablations

    # Analyze factorial experiments
    python scripts/validation/compute_metric_correlations.py --dataset factorials

    # Analyze both
    python scripts/validation/compute_metric_correlations.py --dataset all
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Define metric categories from Phase 4 strategy
METRIC_CATEGORIES = {
    'traditional': ['hypervolume'],
    'trajectory': ['tds'],
    'spatial': ['mce', 'num_modes'],
    'objective': ['pfs'],
    'composite': ['qds']
}

# Flatten to get all metrics
ALL_METRICS = [m for metrics in METRIC_CATEGORIES.values() for m in metrics]


def load_ablation_data() -> pd.DataFrame:
    """
    Load all ablation study results.

    Returns:
        Combined DataFrame with all ablation experiments
    """
    dfs = []
    base_dir = Path('results/ablations')

    # Capacity ablation
    capacity_file = base_dir / 'capacity' / 'all_results.csv'
    if capacity_file.exists():
        df = pd.read_csv(capacity_file)
        df['study'] = 'capacity'
        dfs.append(df)
        print(f"✓ Loaded capacity ablation: {len(df)} experiments")

    # Sampling ablation
    sampling_file = base_dir / 'sampling' / 'all_results.csv'
    if sampling_file.exists():
        df = pd.read_csv(sampling_file)
        df['study'] = 'sampling'
        dfs.append(df)
        print(f"✓ Loaded sampling ablation: {len(df)} experiments")

    # Loss ablation
    loss_file = base_dir / 'loss' / 'base_loss_comparison' / 'results.csv'
    if loss_file.exists():
        df = pd.read_csv(loss_file)
        df['study'] = 'loss'
        dfs.append(df)
        print(f"✓ Loaded loss ablation: {len(df)} experiments")

    if not dfs:
        raise ValueError("No ablation data found!")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined ablation data: {len(combined)} total experiments")

    return combined


def load_factorial_data() -> pd.DataFrame:
    """
    Load all factorial experiment results.

    Returns:
        Combined DataFrame with all factorial experiments
    """
    dfs = []
    base_dir = Path('results/factorials')

    # Define factorial experiments
    factorial_types = ['capacity_sampling', 'capacity_loss', 'sampling_loss']
    tasks = ['hypergrid', 'ngrams', 'molecules', 'sequences']

    for task in tasks:
        for exp_type in factorial_types:
            # Handle directory naming (hypergrid doesn't have task prefix)
            if task == 'hypergrid':
                dir_name = exp_type
            else:
                dir_name = f'{task}_{exp_type}'

            results_file = base_dir / dir_name / 'results.csv'

            if results_file.exists():
                df = pd.read_csv(results_file)
                df['study'] = f'{task}_{exp_type}'
                df['task'] = task
                df['factorial_type'] = exp_type
                dfs.append(df)
                print(f"✓ Loaded {task} {exp_type}: {len(df)} experiments")

    if not dfs:
        raise ValueError("No factorial data found!")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✓ Combined factorial data: {len(combined)} total experiments")

    return combined


def check_metric_availability(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Check which metrics are available in the dataset.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary mapping metric name to availability (True/False)
    """
    availability = {}
    missing_count = 0

    print("\n" + "="*80)
    print("METRIC AVAILABILITY CHECK")
    print("="*80)

    for category, metrics in METRIC_CATEGORIES.items():
        print(f"\n{category.upper()}:")
        for metric in metrics:
            available = metric in df.columns
            if available:
                non_null = df[metric].notna().sum()
                total = len(df)
                pct = (non_null / total * 100) if total > 0 else 0
                status = "✓" if pct > 90 else "⚠️"
                print(f"  {status} {metric:25s}: {non_null}/{total} ({pct:.1f}%)")
            else:
                print(f"  ✗ {metric:25s}: NOT FOUND")
                missing_count += 1

            availability[metric] = available

    print(f"\n{'='*80}")
    print(f"Summary: {len(ALL_METRICS) - missing_count}/{len(ALL_METRICS)} metrics available")

    return availability


def compute_correlation_matrix(df: pd.DataFrame, available_metrics: List[str]) -> pd.DataFrame:
    """
    Compute correlation matrix for available metrics.

    Args:
        df: DataFrame with experiment results
        available_metrics: List of available metric names

    Returns:
        Correlation matrix DataFrame
    """
    # Select only available metrics with sufficient data
    metrics_with_data = []
    for metric in available_metrics:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            if non_null > 10:  # Require at least 10 data points
                metrics_with_data.append(metric)

    if len(metrics_with_data) < 2:
        raise ValueError(f"Insufficient metrics with data (found {len(metrics_with_data)})")

    print(f"\nComputing correlations for {len(metrics_with_data)} metrics...")

    # Compute correlation matrix
    corr_matrix = df[metrics_with_data].corr(method='pearson')

    return corr_matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path, title: str = "Metric Correlations"):
    """
    Create and save correlation heatmap.

    Args:
        corr_matrix: Correlation matrix
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation (r)'},
        ax=ax
    )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save as both PDF and PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap: {output_path}")

    plt.close()


def identify_metric_relationships(corr_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identify highly correlated, independent, and moderately correlated metric pairs.

    Args:
        corr_matrix: Correlation matrix

    Returns:
        Tuple of (redundant_pairs, independent_pairs, moderate_pairs) DataFrames
    """
    # Extract upper triangle (avoid duplicates)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.where(mask)

    # Collect metric pairs
    pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            metric1 = corr_matrix.index[i]
            metric2 = corr_matrix.columns[j]
            r = corr_matrix.iloc[i, j]

            if not np.isnan(r):
                pairs.append({
                    'metric1': metric1,
                    'metric2': metric2,
                    'correlation': r,
                    'abs_correlation': abs(r)
                })

    pairs_df = pd.DataFrame(pairs)

    # Categorize pairs
    redundant = pairs_df[pairs_df['abs_correlation'] > 0.9].sort_values('abs_correlation', ascending=False)
    independent = pairs_df[pairs_df['abs_correlation'] < 0.3].sort_values('abs_correlation')
    moderate = pairs_df[
        (pairs_df['abs_correlation'] >= 0.3) & (pairs_df['abs_correlation'] <= 0.9)
    ].sort_values('abs_correlation', ascending=False)

    return redundant, independent, moderate


def print_correlation_table(corr_matrix: pd.DataFrame):
    """
    Print correlation matrix as a formatted table.

    Args:
        corr_matrix: Correlation matrix DataFrame
    """
    print("\n" + "="*100)
    print("CORRELATION MATRIX (Pearson r)")
    print("="*100)

    # Print header
    metrics = list(corr_matrix.columns)

    # Print table with proper formatting
    # Truncate metric names for table display
    short_names = {m: m[:12] for m in metrics}

    print(f"\n{'Metric':<15s}", end='')
    for m in metrics:
        print(f"{short_names[m]:>8s}", end=' ')
    print()
    print("-" * (15 + 9 * len(metrics)))

    for i, metric1 in enumerate(metrics):
        print(f"{short_names[metric1]:<15s}", end='')
        for j, metric2 in enumerate(metrics):
            r = corr_matrix.iloc[i, j]
            if i == j:
                print(f"{'1.00':>8s}", end=' ')
            elif pd.isna(r):
                print(f"{'---':>8s}", end=' ')
            else:
                # Color code by magnitude
                print(f"{r:>8.2f}", end=' ')
        print()

    print("\nInterpretation:")
    print("  |r| > 0.9 : Very strong correlation (redundant)")
    print("  |r| > 0.7 : Strong correlation")
    print("  |r| > 0.5 : Moderate correlation")
    print("  |r| > 0.3 : Weak correlation")
    print("  |r| ≤ 0.3 : Very weak/independent")


def print_correlation_summary(redundant: pd.DataFrame, independent: pd.DataFrame, moderate: pd.DataFrame):
    """
    Print summary of correlation analysis.

    Args:
        redundant: Highly correlated pairs (|r| > 0.9)
        independent: Independent pairs (|r| < 0.3)
        moderate: Moderately correlated pairs
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*80)

    # Redundant metrics
    print(f"\n🔴 REDUNDANT METRICS (|r| > 0.9): {len(redundant)} pairs")
    if len(redundant) > 0:
        print("\n    These metrics measure very similar things - consider removing redundant ones:\n")
        print(f"    {'Metric 1':<25s} {'Metric 2':<25s} {'Correlation':>12s}")
        print("    " + "-"*65)
        for _, row in redundant.head(15).iterrows():
            print(f"    {row['metric1']:<25s} {row['metric2']:<25s} {row['correlation']:>12.3f}")
        if len(redundant) > 15:
            print(f"    ... and {len(redundant) - 15} more pairs")

    # Independent metrics
    print(f"\n🟢 INDEPENDENT METRICS (|r| < 0.3): {len(independent)} pairs")
    if len(independent) > 0:
        print("\n    These metrics capture different aspects of diversity:\n")
        print(f"    {'Metric 1':<25s} {'Metric 2':<25s} {'Correlation':>12s}")
        print("    " + "-"*65)
        for _, row in independent.head(15).iterrows():
            print(f"    {row['metric1']:<25s} {row['metric2']:<25s} {row['correlation']:>12.3f}")
        if len(independent) > 15:
            print(f"    ... and {len(independent) - 15} more pairs")

    # Moderate correlations (show top 10)
    print(f"\n🟡 MODERATE CORRELATIONS (0.3 ≤ |r| ≤ 0.9): {len(moderate)} pairs")
    if len(moderate) > 0:
        print("\n    Top 10 moderate correlations:\n")
        print(f"    {'Metric 1':<25s} {'Metric 2':<25s} {'Correlation':>12s}")
        print("    " + "-"*65)
        for _, row in moderate.head(10).iterrows():
            print(f"    {row['metric1']:<25s} {row['metric2']:<25s} {row['correlation']:>12.3f}")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    if len(redundant) > 0:
        print(f"⚠️  Found {len(redundant)} redundant metric pairs - consider simplifying metric set")
    else:
        print("✓ No redundant metrics found - all metrics provide unique information")

    if len(independent) > len(redundant):
        print(f"✓ Metrics are largely independent ({len(independent)} independent vs {len(redundant)} redundant)")
    else:
        print(f"⚠️  High redundancy detected ({len(redundant)} redundant vs {len(independent)} independent)")


def save_correlation_summary(redundant: pd.DataFrame, independent: pd.DataFrame,
                            moderate: pd.DataFrame, output_dir: Path):
    """
    Save correlation summary to CSV files.

    Args:
        redundant: Redundant pairs
        independent: Independent pairs
        moderate: Moderate pairs
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    redundant.to_csv(output_dir / 'redundant_metrics.csv', index=False)
    independent.to_csv(output_dir / 'independent_metrics.csv', index=False)
    moderate.to_csv(output_dir / 'moderate_correlations.csv', index=False)

    print(f"\n✓ Saved correlation summaries to: {output_dir}")
    print(f"  - redundant_metrics.csv ({len(redundant)} pairs)")
    print(f"  - independent_metrics.csv ({len(independent)} pairs)")
    print(f"  - moderate_correlations.csv ({len(moderate)} pairs)")


def main():
    parser = argparse.ArgumentParser(
        description='Compute correlation matrices for diversity metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ablations', 'factorials', 'all'],
        default='all',
        help='Dataset to analyze (default: all)'
    )

    args = parser.parse_args()

    print("="*80)
    print("METRIC CORRELATION ANALYSIS")
    print("Validation 1.2: Phase 4 - Metric Validation")
    print("="*80)
    print()

    # Create output directory
    output_dir = Path('results/validation/correlation_matrices')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    datasets_to_analyze = []

    if args.dataset in ['ablations', 'all']:
        try:
            print("Loading ablation studies...")
            ablations_df = load_ablation_data()
            datasets_to_analyze.append(('ablations', ablations_df))
        except Exception as e:
            print(f"⚠️  Could not load ablation data: {e}")

    if args.dataset in ['factorials', 'all']:
        try:
            print("\nLoading factorial experiments...")
            factorials_df = load_factorial_data()
            datasets_to_analyze.append(('factorials', factorials_df))
        except Exception as e:
            print(f"⚠️  Could not load factorial data: {e}")

    if not datasets_to_analyze:
        print("✗ Error: No data could be loaded!")
        sys.exit(1)

    # Analyze each dataset
    for dataset_name, df in datasets_to_analyze:
        print("\n" + "="*80)
        print(f"ANALYZING: {dataset_name.upper()}")
        print("="*80)

        # Check metric availability
        availability = check_metric_availability(df)
        available_metrics = [m for m, avail in availability.items() if avail]

        if len(available_metrics) < 2:
            print(f"⚠️  Insufficient metrics available for {dataset_name} - skipping")
            continue

        # Compute correlation matrix
        corr_matrix = compute_correlation_matrix(df, available_metrics)

        # Plot heatmap
        output_file = output_dir / f'{dataset_name}_correlation_matrix.pdf'
        plot_correlation_heatmap(
            corr_matrix,
            output_file,
            title=f'{dataset_name.capitalize()} - Diversity Metric Correlations'
        )

        # Print correlation table
        print_correlation_table(corr_matrix)

        # Identify relationships
        redundant, independent, moderate = identify_metric_relationships(corr_matrix)

        # Print summary
        print_correlation_summary(redundant, independent, moderate)

        # Save summary
        save_correlation_summary(redundant, independent, moderate,
                                output_dir / dataset_name)

    print("\n" + "="*80)
    print("CORRELATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review correlation heatmaps to identify redundant metrics")
    print("  2. Check redundant_metrics.csv for metrics that could be removed")
    print("  3. Proceed to Factor Analysis (metric_factor_analysis.py)")


if __name__ == '__main__':
    main()
