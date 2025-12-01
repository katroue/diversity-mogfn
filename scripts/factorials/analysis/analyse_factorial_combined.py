#!/usr/bin/env python3
"""
Analyse and plot interaction effects from 2-way factorial experiments across all tasks.

This script creates combined interaction plots showing all four tasks (hypergrid, ngrams,
molecules, sequences) on a single plot with different linestyles for easy comparison.

Supported factorial designs:
    - Capacity × Loss (capacity_loss)
    - Capacity × Sampling (capacity_sampling)
    - Sampling × Loss (sampling_loss)

Tasks:
    - hypergrid (solid lines)
    - ngrams (dashed lines)
    - molecules (dotted lines)
    - sequences (dash-dot lines)

Usage:
    # Basic usage (combines all four tasks):
    python scripts/factorials/analysis/analyse_factorial_combined.py \
        --experiment capacity_loss \
        --metric mce \
        --output results/factorials/analysis/sampling_loss/combined_capacity_loss_mce.pdf \
        --heatmap


    # Specify which tasks to include:
    python scripts/factorials/analysis/analyse_factorial_combined.py \
        --experiment capacity_sampling \
        --metric hypervolume \
        --tasks hypergrid ngrams molecules \
        --output results/factorials/analysis/combined_capacity_sampling_hv.pdf

    # With heatmap:
    python scripts/factorials/analysis/analyse_factorial_combined.py \
        --experiment sampling_loss \
        --metric mce \
        --heatmap

    # Use results_temp.csv instead of results.csv:
    python scripts/factorials/analysis/analyse_factorial_combined.py \
        --experiment capacity_loss \
        --metric tds \
        --use_temp
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# Task configurations
TASK_CONFIGS = {
    'hypergrid': {
        'linestyle': '-',
        'linewidth': 2.5,
        'marker': 'o',
        'label': 'HyperGrid'
    },
    'ngrams': {
        'linestyle': '--',
        'linewidth': 2.5,
        'marker': 's',
        'label': 'N-grams'
    },
    'molecules': {
        'linestyle': ':',
        'linewidth': 2.5,
        'marker': '^',
        'label': 'Molecules'
    },
    'sequences': {
        'linestyle': '-.',
        'linewidth': 2.5,
        'marker': 'D',
        'label': 'Sequences'
    }
}


def get_experiment_info(experiment: str) -> Tuple[str, str]:
    """
    Get factor names from experiment type.

    Args:
        experiment: Experiment type (e.g., 'capacity_loss', 'capacity_sampling')

    Returns:
        (factor_a, factor_b): Tuple of factor column names
    """
    experiment_map = {
        'capacity_loss': ('capacity_level', 'loss_level'),
        'capacity_sampling': ('capacity_level', 'temperature_level'),
        'sampling_loss': ('temperature_level', 'loss_level')
    }

    if experiment not in experiment_map:
        raise ValueError(
            f"Unknown experiment type: {experiment}. "
            f"Supported: {list(experiment_map.keys())}"
        )

    return experiment_map[experiment]


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
        return df

    # Get actual levels present in data (excluding NaN)
    present_levels = df[factor_name].dropna().unique()

    # Filter canonical order to only include present levels
    ordered_levels = [level for level in canonical_order if level in present_levels]

    # Add any levels not in canonical order (at the end), but exclude NaN
    for level in present_levels:
        if level not in ordered_levels and pd.notna(level):
            ordered_levels.append(level)

    # Drop rows with NaN in factor column before creating categorical
    df_clean = df.dropna(subset=[factor_name]).copy()

    # Create categorical with specified order
    df_clean[factor_name] = pd.Categorical(df_clean[factor_name], categories=ordered_levels, ordered=True)

    return df_clean


def load_task_data(experiment: str, task: str, use_temp: bool = False) -> pd.DataFrame:
    """
    Load data for a specific task and experiment.

    Args:
        experiment: Experiment type (e.g., 'capacity_loss')
        task: Task name (e.g., 'hypergrid', 'molecules')
        use_temp: Whether to use results_temp.csv instead of results.csv

    Returns:
        DataFrame with results, or None if file doesn't exist
    """
    results_dir = Path('results/factorials')

    # Construct directory name
    if task == 'hypergrid':
        dir_name = experiment
    else:
        dir_name = f'{task}_{experiment}'

    dir_path = results_dir / dir_name

    # Choose file name
    file_name = 'results_temp.csv' if use_temp else 'results.csv'
    file_path = dir_path / file_name

    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path)
        df['task'] = task  # Add task column for tracking
        return df
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def load_all_tasks(experiment: str, tasks: List[str], use_temp: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load data for all specified tasks.

    Args:
        experiment: Experiment type
        tasks: List of task names to load
        use_temp: Whether to use results_temp.csv

    Returns:
        Dict mapping task name to DataFrame
    """
    data = {}

    for task in tasks:
        df = load_task_data(experiment, task, use_temp)
        if df is not None:
            data[task] = df
            print(f"✓ Loaded {len(df)} rows from {task}")
        else:
            print(f"⚠ No data found for {task}")

    if not data:
        raise ValueError(f"No data found for any task in experiment: {experiment}")

    return data


def create_combined_interaction_plot(
    task_data: Dict[str, pd.DataFrame],
    factor_a: str,
    factor_b: str,
    metric: str,
    output_path: Path
):
    """
    Create combined interaction plot showing all tasks.

    Args:
        task_data: Dict mapping task name to DataFrame
        factor_a: Name of first factor column (x-axis)
        factor_b: Name of second factor column (lines)
        metric: Name of metric to plot (y-axis)
        output_path: Where to save the plot
    """
    # Check if metric exists in all tasks
    for task, df in task_data.items():
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in {task} data")

    # Sort factor levels for all tasks
    for task in task_data:
        task_data[task] = sort_factor_levels(task_data[task].copy(), factor_a)
        task_data[task] = sort_factor_levels(task_data[task], factor_b)

    # Create figure with larger size to accommodate all lines
    fig, ax = plt.subplots(figsize=(14, 8))

    # Get factor B levels (should be same across all tasks)
    first_df = next(iter(task_data.values()))
    factor_b_order = get_factor_order(factor_b)
    if factor_b_order is not None:
        factor_b_levels = [level for level in factor_b_order if level in first_df[factor_b].unique()]
    else:
        factor_b_levels = sorted(first_df[factor_b].unique())

    # Create color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(factor_b_levels)))

    # Plot each task with different linestyle
    for task_name, df in task_data.items():
        task_config = TASK_CONFIGS[task_name]

        for i, level_b in enumerate(factor_b_levels):
            subset = df[df[factor_b] == level_b]

            # Group by factor A and compute mean and std
            grouped = subset.groupby(factor_a, observed=False)[metric].agg(['mean', 'std', 'count'])

            # Compute standard error
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

            # Create label (include factor B level only for first task, task name for all)
            if task_name == list(task_data.keys())[0]:
                label = f"{get_factor_label(factor_b)}: {level_b} ({task_config['label']})"
            else:
                label = f"{level_b} ({task_config['label']})"

            # Plot line with error bars
            ax.errorbar(
                grouped.index,
                grouped['mean'],
                yerr=grouped['se'],
                marker=task_config['marker'],
                linestyle=task_config['linestyle'],
                linewidth=task_config['linewidth'],
                label=label,
                markersize=7,
                capsize=4,
                color=colors[i],
                alpha=0.8
            )

    # Labels and title
    ax.set_xlabel(get_factor_label(factor_a), fontsize=13, fontweight='bold')
    ax.set_ylabel(
        metric.upper() if len(metric) <= 3 else metric.replace('_', ' ').title(),
        fontsize=13,
        fontweight='bold'
    )
    ax.set_title(
        f'{get_factor_label(factor_a)} × {get_factor_label(factor_b)} Interaction\nAcross All Tasks',
        fontsize=15,
        fontweight='bold'
    )

    # Organize legend by columns (one per task)
    ax.legend(
        fontsize=9,
        ncol=len(task_data),
        loc='best',
        framealpha=0.9
    )
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined interaction plot to: {output_path}")

    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to: {png_path}")

    plt.close()


def create_combined_summary_table(
    task_data: Dict[str, pd.DataFrame],
    factor_a: str,
    factor_b: str,
    metric: str,
    output_dir: Path
):
    """
    Create summary statistics table for all tasks.

    Args:
        task_data: Dict mapping task name to DataFrame
        factor_a: Name of first factor
        factor_b: Name of second factor
        metric: Metric to summarize
        output_dir: Directory to save summary
    """
    all_summaries = []

    for task_name, df in task_data.items():
        # Sort factor levels
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
        summary.index = [f"{task_name}_{idx}" for idx in summary.index]

        all_summaries.append(summary)

    # Combine all summaries
    combined = pd.concat(all_summaries)

    # Save to CSV
    summary_path = output_dir / f'combined_summary_{metric}.csv'
    combined.to_csv(summary_path)
    print(f"✓ Saved combined summary table to: {summary_path}")

    # Print to console
    print(f"\nCombined Summary: {metric} (mean ± std)")
    print("=" * 100)
    print(combined)
    print("=" * 100)


def create_combined_heatmap(
    task_data: Dict[str, pd.DataFrame],
    factor_a: str,
    factor_b: str,
    metric: str,
    output_path: Path
):
    """
    Create side-by-side heatmaps for all tasks.

    Args:
        task_data: Dict mapping task name to DataFrame
        factor_a: Name of first factor
        factor_b: Name of second factor
        metric: Metric to plot
        output_path: Where to save the plot
    """
    num_tasks = len(task_data)
    fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 5))

    if num_tasks == 1:
        axes = [axes]

    for idx, (task_name, df) in enumerate(task_data.items()):
        # Sort factor levels
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
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            ax=axes[idx],
            cbar_kws={'label': metric.upper() if len(metric) <= 3 else metric.replace('_', ' ').title()}
        )

        axes[idx].set_xlabel(get_factor_label(factor_a), fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(get_factor_label(factor_b), fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{TASK_CONFIGS[task_name]["label"]}', fontsize=13, fontweight='bold')

    # Overall title
    fig.suptitle(
        f'{metric.upper()} Heatmaps Across Tasks',
        fontsize=15,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    # Save
    heatmap_path = output_path.parent / f'combined_heatmap_{metric}.pdf'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined heatmap to: {heatmap_path}")

    # PNG version
    png_path = heatmap_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()


def create_small_multiples_grid(
    task_data: Dict[str, pd.DataFrame],
    factor_a: str,
    factor_b: str,
    metric: str,
    output_path: Path
):
    """
    Create 2×2 grid of heatmaps for all tasks with consistent color scale.

    Args:
        task_data: Dict mapping task name to DataFrame
        factor_a: Name of first factor
        factor_b: Name of second factor
        metric: Metric to plot
        output_path: Where to save the plot
    """
    # Determine grid size based on number of tasks
    num_tasks = len(task_data)
    if num_tasks <= 2:
        nrows, ncols = 1, num_tasks
        figsize = (7 * ncols, 6)
    elif num_tasks <= 4:
        nrows, ncols = 2, 2
        figsize = (14, 12)
    else:
        nrows = (num_tasks + 2) // 3
        ncols = 3
        figsize = (7 * ncols, 6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easier indexing
    if num_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Compute global min/max for consistent color scale
    all_pivots = []
    task_names_ordered = ['hypergrid', 'ngrams', 'molecules', 'sequences']
    task_names_ordered = [t for t in task_names_ordered if t in task_data]

    for task_name in task_names_ordered:
        df = task_data[task_name]
        # Sort factor levels
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
        all_pivots.append(pivot)

    # Find global min/max
    all_values = pd.concat([p.stack() for p in all_pivots])
    vmin, vmax = all_values.min(), all_values.max()

    # Create heatmaps
    for idx, task_name in enumerate(task_names_ordered):
        pivot = all_pivots[idx]

        # Create heatmap with shared colorbar scale
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            ax=axes[idx],
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': metric.upper() if len(metric) <= 3 else metric.replace('_', ' ').title()},
            annot_kws={'fontsize': 10}
        )

        axes[idx].set_xlabel(get_factor_label(factor_a), fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(get_factor_label(factor_b), fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{TASK_CONFIGS[task_name]["label"]}', fontsize=13, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(task_names_ordered), len(axes)):
        axes[idx].axis('off')

    # Overall title
    metric_title = metric.upper() if len(metric) <= 3 else metric.replace('_', ' ').title()
    fig.suptitle(
        f'{get_factor_label(factor_a)} × {get_factor_label(factor_b)} Interaction\n{metric_title} Across All Tasks',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save
    grid_path = output_path.parent / f'grid_{metric}_{factor_a}_{factor_b}.pdf'
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved small multiples grid to: {grid_path}")

    # PNG version
    png_path = grid_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved PNG version to: {png_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze factorial experiments across all tasks (hypergrid, ngrams, molecules, sequences)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['capacity_loss', 'capacity_sampling', 'sampling_loss'],
        help='Experiment type to analyze'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file for interaction plot (default: results/factorials/analysis/combined_<experiment>_<metric>.pdf)'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='mce',
        help='Metric to plot (default: mce). Options: mce, hypervolume, tds, qds, etc.'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=['hypergrid', 'ngrams', 'molecules', 'sequences'],
        choices=['hypergrid', 'ngrams', 'molecules', 'sequences'],
        help='Tasks to include in analysis (default: all four tasks)'
    )

    parser.add_argument(
        '--use_temp',
        action='store_true',
        help='Use results_temp.csv instead of results.csv'
    )

    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Also create combined heatmap visualizations'
    )

    parser.add_argument(
        '--grid',
        action='store_true',
        help='Create small multiples grid (2×2 layout of heatmaps for all tasks)'
    )

    args = parser.parse_args()

    print(f"\nAnalyzing factorial experiment: {args.experiment}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Metric: {args.metric}")
    print(f"Using: {'results_temp.csv' if args.use_temp else 'results.csv'}")

    # Get factor names
    factor_a, factor_b = get_experiment_info(args.experiment)
    print(f"Factors: {get_factor_label(factor_a)} × {get_factor_label(factor_b)}")

    # Load data for all tasks
    print(f"\nLoading data...")
    task_data = load_all_tasks(args.experiment, args.tasks, args.use_temp)

    if not task_data:
        print("Error: No data loaded for any task!")
        sys.exit(1)

    # Set output path
    if args.output is None:
        output_dir = Path('results/factorials/analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f'combined_{args.experiment}_{args.metric}.pdf'

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create combined interaction plot
    print(f"\nCreating combined interaction plot...")
    try:
        create_combined_interaction_plot(
            task_data,
            factor_a,
            factor_b,
            args.metric,
            args.output
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create combined summary table
    print(f"\nCreating combined summary table...")
    create_combined_summary_table(
        task_data,
        factor_a,
        factor_b,
        args.metric,
        args.output.parent
    )

    # Create combined heatmap if requested
    if args.heatmap:
        print(f"\nCreating combined heatmap...")
        create_combined_heatmap(
            task_data,
            factor_a,
            factor_b,
            args.metric,
            args.output
        )

    # Create small multiples grid if requested
    if args.grid:
        print(f"\nCreating small multiples grid (2×2 layout)...")
        create_small_multiples_grid(
            task_data,
            factor_a,
            factor_b,
            args.metric,
            args.output
        )

    print(f"\n✓ Combined analysis complete!")


if __name__ == '__main__':
    main()