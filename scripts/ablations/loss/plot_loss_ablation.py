#!/usr/bin/env python3
"""
Create 2x2 grid plot for loss ablation experiment groups.

Shows all 5 seeds as individual points with mean ± std prominently displayed.
Uses violin plots (no bar charts) for distribution visualization.

Usage:
    python scripts/ablations/loss/plot_loss_ablation.py \
        --experiment_group base_loss_comparison \
        --metrics mce pfs qds hypervolume

    python scripts/ablations/loss/plot_loss_ablation.py \
        --experiment_group base_loss_comparison \
        --metrics mce hypervolume qds pfs \
        --output results/ablations/loss/figures/base_loss_4metrics.pdf
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Experiment group mappings from config
EXPERIMENT_GROUPS = {
    'base_loss_comparison': {
        'experiments': [
            'flow_matching',
            'detailed_balance',
            'trajectory_balance',
            'subtrajectory_balance_05',
            'subtrajectory_balance_09',
            'subtrajectory_balance_095',
        ],
        'title': 'Base Loss Function Comparison',
        'xlabel': 'Loss Function',
        'display_names': {
            'flow_matching': 'FM',
            'detailed_balance': 'DB',
            'trajectory_balance': 'TB',
            'subtrajectory_balance_05': 'SubTB(0.5)',
            'subtrajectory_balance_09': 'SubTB(0.9)',
            'subtrajectory_balance_095': 'SubTB(0.95)',
        }
    },
    'entropy_regularization': {
        'experiments': [
            'none',
            'entropy_001',
            'entropy_005',
            'entropy_01',
            'entropy_05'
        ],
        'title': 'Entropy Regularization Effects',
        'xlabel': 'Entropy Regularization (β)',
        'display_names': {
            'none': 'None',
            'entropy_001': 'β=0.01',
            'entropy_005': 'β=0.05',
            'entropy_01': 'β=0.1',
            'entropy_05': 'β=0.5'
        }
    },
    'kl_regularization': {
        'experiments': [
            'none',
            'kl_uniform_001',
            'kl_uniform_01'
        ],
        'title': 'KL Regularization to Uniform',
        'xlabel': 'KL Regularization',
        'display_names': {
            'none': 'None',
            'kl_uniform_001': 'β=0.01',
            'kl_uniform_01': 'β=0.1'
        }
    },
    'loss_modifications': {
        'experiments': [
            'standard',
            'temperature_scaled_logits',
            'reward_shaping_diversity'
        ],
        'title': 'Loss Modifications',
        'xlabel': 'Modification Type',
        'display_names': {
            'standard': 'Standard',
            'temperature_scaled_logits': 'Temp-Scale',
            'reward_shaping_diversity': 'Reward-Shape'
        }
    }
}


def load_data(results_dir):
    """Load loss ablation results from directory structure."""
    results_dir = Path(results_dir)

    # Check for results.csv first (has all seeds - preferred)
    results_csv = results_dir / 'results.csv'
    if results_csv.exists():
        print(f"Loading from results.csv: {results_csv}")
        df = pd.read_csv(results_csv)

        # Validate data - check for duplicates
        exp_base = df['exp_name'].str.replace(r'_seed\d+$', '', regex=True)
        seed_counts = df.groupby(exp_base)['seed'].nunique()
        max_seeds = seed_counts.max()

        if max_seeds > 5:
            print(f"  WARNING: results.csv has duplicates (max {max_seeds} seeds per config)")
            print(f"  Falling back to summary.csv for reliable data")
        else:
            print(f"  Loaded {len(df)} rows with per-seed data")
            return df

    # Fall back to summary CSV if results.csv doesn't exist or is corrupted
    summary_csv = results_dir / 'summary.csv'
    if summary_csv.exists():
        print(f"Loading from summary: {summary_csv}")
        print(f"  Warning: Using aggregated data (no individual seeds)")
        df = pd.read_csv(summary_csv)
        # Extract configuration name from 'configuration' column
        if 'configuration' in df.columns:
            df['exp_name'] = df['configuration']

        # Mark as aggregated data
        df['_is_aggregated'] = True
        return df

    # Otherwise load from individual experiment directories
    all_results = []

    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metrics_file = exp_dir / 'metrics.json'
        if not metrics_file.exists():
            continue

        # Load metrics
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Extract exp_name and seed from directory name
        exp_name = exp_dir.name

        # Add to results
        metrics['exp_name'] = exp_name
        all_results.append(metrics)

    if not all_results:
        raise ValueError(f"No results found in {results_dir}")

    df = pd.DataFrame(all_results)
    return df


def extract_base_name(exp_name):
    """Extract base experiment name without seed suffix."""
    # Remove _seed123 pattern
    import re
    base = re.sub(r'_seed\d+$', '', exp_name)
    return base


def plot_single_metric(ax, exp_df, available_exps, display_names, metric, xlabel, show_legend=False, is_aggregated=False):
    """
    Plot a single metric on the given axis.

    Args:
        ax: Matplotlib axis
        exp_df: DataFrame filtered to experiment group
        available_exps: List of available experiment names
        display_names: Dict mapping exp names to display labels
        metric: Metric to plot
        xlabel: X-axis label
        show_legend: Whether to show legend (only on first subplot)
        is_aggregated: Whether data is already aggregated (has _mean, _std columns)
    """
    # Handle aggregated data format
    if is_aggregated:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'

        if mean_col not in exp_df.columns:
            return None

        # Extract mean and std for each experiment
        summary_data = []
        for exp in available_exps:
            exp_row = exp_df[exp_df['exp_base'] == exp]
            if len(exp_row) > 0:
                summary_data.append({
                    'exp_base': exp,
                    'mean': exp_row[mean_col].iloc[0],
                    'std': exp_row[std_col].iloc[0],
                    'count': exp_row.get('num_seeds', pd.Series([5])).iloc[0]
                })
        summary = pd.DataFrame(summary_data).set_index('exp_base')
    else:
        # Calculate summary statistics from raw data
        summary = exp_df.groupby('exp_base', observed=False)[metric].agg(['mean', 'std', 'count'])

    # 1. Background: Violin plot showing distribution (skip for aggregated data)
    if not is_aggregated:
        metric_col = metric
        parts = ax.violinplot(
            [exp_df[exp_df['exp_base'] == exp][metric_col].values for exp in available_exps],
            positions=range(len(available_exps)),
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        # Style violins with light blue
        for pc in parts['bodies']:
            pc.set_facecolor('#87CEEB')
            pc.set_alpha(0.3)
            pc.set_edgecolor('#4682B4')
            pc.set_linewidth(1.5)

        # 2. Individual seed points (medium size, colorful, slightly transparent)
        colors = plt.cm.Set2(np.linspace(0, 1, 5))
        seeds = sorted(exp_df['seed'].unique()) if 'seed' in exp_df.columns else [0]

        for seed_idx, seed in enumerate(seeds):
            if 'seed' in exp_df.columns:
                seed_data = exp_df[exp_df['seed'] == seed].sort_values('exp_base')
            else:
                seed_data = exp_df.sort_values('exp_base')

            x_positions = [available_exps.index(exp) for exp in seed_data['exp_base'] if exp in available_exps]
            if len(x_positions) == 0:
                continue

            # Add small jitter for visibility
            x_jitter = np.array(x_positions) + np.random.normal(0, 0.05, len(x_positions))

            y_values = seed_data[seed_data['exp_base'].isin(available_exps)][metric_col].values
            if len(y_values) > 0:
                ax.scatter(x_jitter, y_values,
                          s=60, alpha=0.7, c=[colors[seed_idx % len(colors)]],
                          edgecolors='black', linewidths=0.8,
                          label=f'Seed {seed}' if show_legend else '', zorder=5)
    else:
        # For aggregated data, just show bars indicating std range
        for idx, exp in enumerate(available_exps):
            if exp in summary.index:
                mean = summary.loc[exp, 'mean']
                std = summary.loc[exp, 'std']
                # Draw a light box showing mean ± std range
                ax.add_patch(plt.Rectangle((idx - 0.3, mean - std), 0.6, 2 * std,
                                          facecolor='lightblue', alpha=0.3,
                                          edgecolor='#4682B4', linewidth=1.5, zorder=1))

    # 3. Prominent mean with error bars
    x_pos = range(len(available_exps))
    means = [summary.loc[exp, 'mean'] if exp in summary.index else np.nan for exp in available_exps]
    stds = [summary.loc[exp, 'std'] if exp in summary.index else 0 for exp in available_exps]

    ax.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt='D',
        markersize=10,
        capsize=5,
        capthick=2,
        color='darkred',
        ecolor='darkred',
        linewidth=2,
        markeredgecolor='black',
        markeredgewidth=1.5,
        label='Mean ± Std' if show_legend else '',
        zorder=10
    )

    # Formatting
    ax.set_xticks(x_pos)
    labels = [display_names.get(exp, exp) for exp in available_exps]
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold', rotation=45, ha='right')
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(metric.upper(), fontsize=12, fontweight='bold', pad=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax.set_axisbelow(True)

    return summary


def create_four_panel_plot(df, experiment_group, metrics, output_path=None):
    """
    Create a 2x2 grid plot with four metrics.

    Args:
        df: DataFrame with results
        experiment_group: One of the keys in EXPERIMENT_GROUPS
        metrics: List of 4 metrics to plot
        output_path: Optional path to save figure
    """
    if experiment_group not in EXPERIMENT_GROUPS:
        raise ValueError(f"Unknown experiment_group: {experiment_group}. "
                        f"Must be one of: {list(EXPERIMENT_GROUPS.keys())}")

    if len(metrics) != 4:
        raise ValueError(f"Must provide exactly 4 metrics, got {len(metrics)}")

    # Get experiment configuration
    exp_config = EXPERIMENT_GROUPS[experiment_group]
    exp_list = exp_config['experiments']
    title = exp_config['title']
    xlabel = exp_config['xlabel']
    display_names = exp_config['display_names']

    # Extract base experiment name
    df['exp_base'] = df['exp_name'].apply(extract_base_name)

    # Filter data to this experiment group
    exp_df = df[df['exp_base'].isin(exp_list)].copy()

    if len(exp_df) == 0:
        raise ValueError(f"No data found for experiment_group: {experiment_group}")

    # Filter to only available experiments
    available_exps = [exp for exp in exp_list if exp in exp_df['exp_base'].values]

    if len(available_exps) == 0:
        raise ValueError(f"No matching experiments found in data for {experiment_group}")

    # Create categorical ordering
    exp_df['exp_base'] = pd.Categorical(exp_df['exp_base'],
                                        categories=available_exps,
                                        ordered=True)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Set style
    sns.set_palette("husl")

    # Check if data is aggregated
    is_aggregated = '_is_aggregated' in df.columns and df['_is_aggregated'].iloc[0]

    # Plot each metric
    summaries = {}
    for idx, metric in enumerate(metrics):
        # Check if metric exists (either raw or aggregated format)
        metric_exists = metric in df.columns or f'{metric}_mean' in df.columns

        if not metric_exists:
            print(f"Warning: Metric '{metric}' not found in data. Skipping.")
            axes[idx].text(0.5, 0.5, f"Metric '{metric}'\nnot found",
                         ha='center', va='center', fontsize=14,
                         transform=axes[idx].transAxes)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue

        # Plot on subplot (show legend only on first subplot)
        summary = plot_single_metric(axes[idx], exp_df, available_exps,
                                     display_names, metric, xlabel,
                                     show_legend=(idx == 0),
                                     is_aggregated=is_aggregated)
        if summary is not None:
            summaries[metric] = summary

        # Add panel label
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        axes[idx].text(-0.08, 1.05, panel_labels[idx],
                      transform=axes[idx].transAxes,
                      fontsize=16, fontweight='bold', va='top')

    # Add legend to first subplot
    if len(summaries) > 0:
        axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1),
                      frameon=True, shadow=True, fontsize=9,
                      title='Seeds & Statistics', title_fontsize=10)

    # Overall title
    fig.suptitle(f'{title}\n({", ".join([m.upper() for m in metrics])})',
                 fontsize=16, fontweight='bold', y=0.995)

    # Adjust spacing
    plt.subplots_adjust(hspace=0.40, wspace=0.3, top=0.93, bottom=0.05, left=0.08, right=0.95)

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        # Also save PNG version
        png_path = output_path.with_suffix('.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')

        print(f"✓ Saved: {output_path}")
        print(f"✓ Saved: {png_path}")
    else:
        plt.show()

    plt.close()

    # Print summary statistics for all metrics
    print(f"\n{'='*80}")
    print(f"{title}: Summary Statistics")
    print(f"{'='*80}")

    for metric, summary in summaries.items():
        print(f"\n{metric.upper()}:")
        print(summary.to_string())
        if len(summary) > 0:
            best_exp = summary['mean'].idxmax()
            best_mean = summary.loc[best_exp, 'mean']
            best_std = summary.loc[best_exp, 'std']
            best_display = display_names.get(best_exp, best_exp)
            print(f"  → Best: {best_display} ({best_mean:.4f} ± {best_std:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Create 2x2 grid plot with 4 metrics for loss ablation experiment group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Base loss function comparison
  python scripts/ablations/loss/plot_loss_ablation.py \\
      --experiment_group base_loss_comparison \\
      --metrics mce pfs qds hypervolume

  # Entropy regularization effects
  python scripts/ablations/loss/plot_loss_ablation.py \\
      --experiment_group entropy_regularization \\
      --metrics mce hypervolume qds fci \\
      --output results/ablations/loss/figures/entropy_4metrics.pdf

  # KL regularization
  python scripts/ablations/loss/plot_loss_ablation.py \\
      -g kl_regularization \\
      -m mce pfs qds hypervolume \\
      -o results/ablations/loss/figures/kl_4metrics.pdf

Available experiment groups:
  - base_loss_comparison: TB, DB, SubTB(0.5, 0.9, 0.95), FM
  - entropy_regularization: Different β values for entropy regularization
  - kl_regularization: KL divergence to uniform policy
  - loss_modifications: Temperature scaling, reward shaping

Common metrics:
  - hypervolume, spacing, spread (traditional MO metrics)
  - tds, mpd (trajectory diversity)
  - mce, pmd (spatial diversity)
  - pfs (Pareto front smoothness)
  - qds (quality-diversity score)
  - fci (flow concentration)
  - rbd (replay buffer diversity)

Recommended metric combinations:
  base_loss_comparison:     mce pfs qds hypervolume
  entropy_regularization:   mce hypervolume qds fci
  kl_regularization:        mce pfs qds hypervolume
  loss_modifications:       mce qds pfs hypervolume
        """
    )

    parser.add_argument('--experiment_group', '-g', type=str, required=True,
                       choices=list(EXPERIMENT_GROUPS.keys()),
                       help='Experiment group to plot')

    parser.add_argument('--metrics', '-m', type=str, nargs=4, required=True,
                       help='Four metrics to plot (space-separated)')

    parser.add_argument('--results_dir', '-r', type=str,
                       default=None,
                       help='Path to results directory (auto-detected from experiment_group if not provided)')

    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot (PDF). If not provided, saves to default location.')

    args = parser.parse_args()

    # Auto-detect results directory based on experiment group if not provided
    if args.results_dir is None:
        args.results_dir = f'results/ablations/loss/{args.experiment_group}'
        print(f"Auto-detected results directory: {args.results_dir}")

    # Load data
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Loading data from: {results_dir}")
    df = load_data(results_dir)
    print(f"Loaded {len(df)} experiments")

    # Set default output path if not provided
    if args.output is None:
        output_dir = Path('results/ablations/loss/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_str = '_'.join(args.metrics)
        args.output = str(output_dir / f'{args.experiment_group}_{metrics_str}.pdf')
        print(f"No output path specified, using default: {args.output}")

    # Create plot
    create_four_panel_plot(df, args.experiment_group, args.metrics, args.output)


if __name__ == '__main__':
    main()
