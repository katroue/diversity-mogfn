#!/usr/bin/env python3
"""
Analyse and plot interaction effects from 2-way factorial experiments.

This script creates interaction plots for 2-way factorial experiments,
automatically detecting the factor types from the CSV columns.
Works for any environment (hypergrid, ngrams, molecules and sequences).

Supported factorial designs:
    - Capacity × Temperature (capacity_level, temperature_level)
    - Capacity × Loss (capacity_level, loss_level)
    - Temperature × Loss (temperature_level, loss_level)

Factor levels are shown in canonical order:
    - capacity_level: small, medium, large, xlarge
    - temperature_level: low, high, very_high
    - loss_level: tb, subtb, subtb_entropy

Usage:
    # Basic usage (auto-detects factorial design):
    python scripts/factorials/hypergrid/analyse_factorial.py \
        --input results/factorials/capacity_loss/results.csv \
        --metric mce

    # With heatmap:
    python scripts/factorials/hypergrid/analyse_factorial.py \
        --input results/factorials/capacity_loss/results.csv \
        --metric hypervolume \
        --heatmap

    # Specify output location:
    python scripts/factorials/hypergrid/analyse_factorial.py \
        --input results/factorials/sampling_loss/results.csv \
        --metric mce \
        --output results/factorials/sampling_loss/interaction_plot_mce.pdf

    # Works for any environment:
    python scripts/factorials/ngrams/analyse_factorial.py \
        --input results/factorials/ngrams_sampling_loss/results.csv \
        --metric tds
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def detect_factorial_design(df: pd.DataFrame) -> tuple:
    """
    Detect which factorial design is being used based on column names.

    Args:
        df: DataFrame with factorial results

    Returns:
        (factor_a, factor_b): Tuple of factor column names

    Raises:
        ValueError: If factorial design cannot be detected
    """
    # Possible factor column names
    possible_factors = ['capacity_level', 'temperature_level', 'loss_level']

    # Find which factors are present
    present_factors = [f for f in possible_factors if f in df.columns]

    if len(present_factors) < 2:
        raise ValueError(
            f"Could not detect 2-way factorial design. "
            f"Found factor columns: {present_factors}. "
            f"Expected 2 of: {possible_factors}"
        )

    if len(present_factors) > 2:
        print(f"Warning: Found {len(present_factors)} factors, using first 2: {present_factors[:2]}")

    return present_factors[0], present_factors[1]


def get_factor_label(factor_name: str) -> str:
    """Get human-readable label for factor."""
    labels = {
        'capacity_level': 'Model Capacity',
        'temperature_level': 'Sampling Temperature',
        'loss_level': 'Loss Function'
    }
    return labels.get(factor_name, factor_name)


def get_factor_order(factor_name: str) -> list:
    """
    Get the canonical ordering for factor levels.

    Args:
        factor_name: Name of the factor (e.g., 'capacity_level')

    Returns:
        List of level names in the desired order
    """
    orders = {
        'capacity_level': ['small', 'medium', 'large', 'xlarge'],
        'temperature_level': ['low', 'high', 'very_high'],
        'loss_level': ['tb', 'subtb', 'subtb_entropy']
    }
    return orders.get(factor_name, None)


def sort_factor_levels(df: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """
    Sort DataFrame by factor levels in canonical order.

    Args:
        df: DataFrame to sort
        factor_name: Name of factor column to sort by

    Returns:
        Sorted DataFrame
    """
    canonical_order = get_factor_order(factor_name)

    if canonical_order is None:
        # No canonical order defined, use existing order
        return df

    # Get actual levels present in data
    present_levels = df[factor_name].unique()

    # Filter canonical order to only include present levels
    ordered_levels = [level for level in canonical_order if level in present_levels]

    # Add any levels not in canonical order (at the end)
    for level in present_levels:
        if level not in ordered_levels:
            ordered_levels.append(level)

    # Create categorical with specified order
    df[factor_name] = pd.Categorical(df[factor_name], categories=ordered_levels, ordered=True)

    return df


def create_interaction_plot(df: pd.DataFrame,
                           factor_a: str,
                           factor_b: str,
                           metric: str,
                           output_path: Path):
    """
    Create interaction plot for 2-way factorial experiment.

    Args:
        df: DataFrame with factorial results
        factor_a: Name of first factor column (x-axis)
        factor_b: Name of second factor column (lines)
        metric: Name of metric to plot (y-axis)
        output_path: Where to save the plot
    """
    # Check if metric exists
    if metric not in df.columns:
        available_metrics = [c for c in df.columns if c not in
                           ['exp_name', 'condition_name', 'seed', 'num_parameters',
                            'training_time', 'final_loss'] and '_level' not in c]
        raise ValueError(
            f"Metric '{metric}' not found in DataFrame. "
            f"Available metrics: {available_metrics}"
        )

    # Sort factor levels in canonical order
    df = sort_factor_levels(df.copy(), factor_a)
    df = sort_factor_levels(df, factor_b)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique levels in sorted order
    factor_b_order = get_factor_order(factor_b)
    if factor_b_order is not None:
        # Use canonical order
        factor_b_levels = [level for level in factor_b_order if level in df[factor_b].unique()]
    else:
        # Fallback to sorted
        factor_b_levels = sorted(df[factor_b].unique())

    # Plot interaction lines
    colors = plt.cm.Set2(np.linspace(0, 1, len(factor_b_levels)))

    for i, level_b in enumerate(factor_b_levels):
        subset = df[df[factor_b] == level_b]

        # Group by factor A and compute mean and std
        grouped = subset.groupby(factor_a, observed=False)[metric].agg(['mean', 'std', 'count'])

        # Compute standard error
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

        # Plot line with error bars
        ax.errorbar(
            grouped.index,
            grouped['mean'],
            yerr=grouped['se'],
            marker='o',
            label=f'{get_factor_label(factor_b)}: {level_b}',
            linewidth=2,
            markersize=8,
            capsize=5,
            color=colors[i]
        )

    # Labels and title
    ax.set_xlabel(get_factor_label(factor_a), fontsize=12)
    ax.set_ylabel(metric.upper() if len(metric) <= 3 else metric.replace('_', ' ').title(),
                 fontsize=12)
    ax.set_title(f'{get_factor_label(factor_a)} × {get_factor_label(factor_b)} Interaction',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved interaction plot to: {output_path}")

    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to: {png_path}")

    plt.close()


def create_summary_table(df: pd.DataFrame,
                        factor_a: str,
                        factor_b: str,
                        metric: str,
                        output_dir: Path):
    """
    Create summary statistics table.

    Args:
        df: DataFrame with factorial results
        factor_a: Name of first factor
        factor_b: Name of second factor
        metric: Metric to summarize
        output_dir: Directory to save summary
    """
    # Sort factor levels in canonical order
    df = sort_factor_levels(df.copy(), factor_a)
    df = sort_factor_levels(df, factor_b)

    # Create pivot table with mean values
    pivot_mean = df.pivot_table(
        values=metric,
        index=factor_a,
        columns=factor_b,
        aggfunc='mean',
        observed=False
    )

    # Create pivot table with std values
    pivot_std = df.pivot_table(
        values=metric,
        index=factor_a,
        columns=factor_b,
        aggfunc='std',
        observed=False
    )

    # Combine mean ± std
    summary = pivot_mean.round(4).astype(str) + ' ± ' + pivot_std.round(4).astype(str)

    # Save to CSV
    summary_path = output_dir / f'summary_{metric}.csv'
    summary.to_csv(summary_path)
    print(f"✓ Saved summary table to: {summary_path}")

    # Print to console
    print(f"\nSummary: {metric} (mean ± std)")
    print("=" * 80)
    print(summary)
    print("=" * 80)


def create_heatmap(df: pd.DataFrame,
                  factor_a: str,
                  factor_b: str,
                  metric: str,
                  output_path: Path):
    """
    Create heatmap of mean metric values.

    Args:
        df: DataFrame with factorial results
        factor_a: Name of first factor
        factor_b: Name of second factor
        metric: Metric to plot
        output_path: Where to save the plot
    """
    # Sort factor levels in canonical order
    df = sort_factor_levels(df.copy(), factor_a)
    df = sort_factor_levels(df, factor_b)

    # Create pivot table
    pivot = df.pivot_table(
        values=metric,
        index=factor_b,
        columns=factor_a,
        aggfunc='mean',
        observed=False
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        ax=ax,
        cbar_kws={'label': metric.upper() if len(metric) <= 3 else metric.replace('_', ' ').title()}
    )

    ax.set_xlabel(get_factor_label(factor_a), fontsize=12)
    ax.set_ylabel(get_factor_label(factor_b), fontsize=12)
    ax.set_title(f'{metric.upper()} Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    heatmap_path = output_path.parent / f'heatmap_{metric}.pdf'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {heatmap_path}")

    # PNG version
    png_path = heatmap_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze factorial experiments for Multi-Objective GFlowNets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to input CSV file with factorial results'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file for interaction plot (default: <input_dir>/interaction_plot.pdf)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='mce',
        help='Metric to plot (default: mce). Options: mce, hypervolume, tds, qds, etc.'
    )

    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Also create a heatmap visualization'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"✓ Loaded {len(df)} rows")

    # Detect factorial design
    print("\nDetecting factorial design...")
    try:
        factor_a, factor_b = detect_factorial_design(df)
        print(f"✓ Detected 2-way factorial: {get_factor_label(factor_a)} × {get_factor_label(factor_b)}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Set output path
    if args.output is None:
        args.output = args.input.parent / 'interaction_plot.pdf'

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create interaction plot
    print(f"\nCreating interaction plot for metric: {args.metric}")
    try:
        create_interaction_plot(df, factor_a, factor_b, args.metric, args.output)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create summary table
    print(f"\nCreating summary table...")
    create_summary_table(df, factor_a, factor_b, args.metric, args.output.parent)

    # Create heatmap if requested
    if args.heatmap:
        print(f"\nCreating heatmap...")
        create_heatmap(df, factor_a, factor_b, args.metric, args.output)

    print(f"\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
