#!/usr/bin/env python3
"""
Create a 2x2 grid plot with four metrics for a specific sampling ablation experiment type.

Shows all 5 seeds as individual points with mean ± std prominently displayed.
Uses violin plots (no bar charts) for distribution visualization.

Usage:
    python scripts/ablations/sampling/plot_experiment_type_sampling_ablation.py \
        --experiment_type policy_type \
        --metrics mce pas der qds

    python scripts/ablations/sampling/plot_experiment_type_sampling_ablation.py \
        --experiment_type policy_type \
        --metrics mce der pas spread \
        --output results/ablations/sampling/report/policy_4metrics.pdf
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Experiment type mappings from config
EXPERIMENT_GROUPS = {
    'temperature': {
        'experiments': ['temp_low', 'temp_medium', 'temp_high', 'temp_very_high'],
        'title': 'Exploration Temperature',
        'xlabel': 'Temperature Setting'
    },
    'sampling_strategy': {
        'experiments': ['greedy', 'categorical', 'top_k', 'top_p'],
        'title': 'Sampling Strategies',
        'xlabel': 'Strategy'
    },
    'policy_type': {
        'experiments': ['on_policy_pure', 'off_policy_10', 'off_policy_25', 'off_policy_50'],
        'title': 'On-Policy vs Off-Policy',
        'xlabel': 'Policy Configuration'
    },
    'preference_diversity': {
        'experiments': ['pref_uniform', 'pref_dirichlet_low', 'pref_dirichlet_medium', 'pref_dirichlet_high'],
        'title': 'Preference Diversity',
        'xlabel': 'Preference Distribution'
    },
    'batch_size': {
        'experiments': ['batch_32', 'batch_64', 'batch_256', 'batch_512'],
        'title': 'Batch Size Effects',
        'xlabel': 'Batch Size'
    },
    'combined': {
        'experiments': ['diverse_sampling', 'quality_sampling'],
        'title': 'Combined Strategies',
        'xlabel': 'Strategy'
    }
}


def load_data(results_path):
    """Load results and extract base experiment names."""
    df = pd.read_csv(results_path)

    # Extract base experiment name (without seed suffix)
    df['exp_base'] = df['exp_name'].str.replace(r'_seed\d+$', '', regex=True)

    return df


def plot_single_metric(ax, exp_df, available_exps, metric, xlabel, show_legend=False):
    """
    Plot a single metric on the given axis.

    Args:
        ax: Matplotlib axis
        exp_df: DataFrame filtered to experiment type
        available_exps: List of available experiment names
        metric: Metric to plot
        xlabel: X-axis label
        show_legend: Whether to show legend (only on first subplot)
    """
    # Calculate summary statistics
    summary = exp_df.groupby('exp_base', observed=False)[metric].agg(['mean', 'std', 'count'])

    # 1. Background: Violin plot showing distribution
    parts = ax.violinplot(
        [exp_df[exp_df['exp_base'] == exp][metric].values for exp in available_exps],
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
    for seed_idx, seed in enumerate(sorted(exp_df['seed'].unique())):
        seed_data = exp_df[exp_df['seed'] == seed].sort_values('exp_base')
        x_positions = [available_exps.index(exp) for exp in seed_data['exp_base']]

        # Add small jitter for visibility
        x_jitter = np.array(x_positions) + np.random.normal(0, 0.05, len(x_positions))

        ax.scatter(x_jitter, seed_data[metric].values,
                  s=60, alpha=0.7, c=[colors[seed_idx]],
                  edgecolors='black', linewidths=0.8,
                  label=f'Seed {seed}' if show_legend else '', zorder=5)

    # 3. Prominent mean with error bars
    x_pos = range(len(available_exps))
    ax.errorbar(
        x_pos,
        summary['mean'],
        yerr=summary['std'],
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
    ax.set_xticklabels([exp.replace('_', '\n') for exp in available_exps],
                       fontsize=9, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(metric.upper(), fontsize=12, fontweight='bold', pad=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax.set_axisbelow(True)

    # Add panel label (a, b, c, d)
    panel_idx = ['(a)', '(b)', '(c)', '(d)']
    # Get subplot position from ax
    for idx, label in enumerate(panel_idx):
        break  # Will be set externally

    return summary


def create_four_panel_plot(df, experiment_type, metrics, output_path=None):
    """
    Create a 2x2 grid plot with four metrics.

    Args:
        df: DataFrame with results
        experiment_type: One of the keys in EXPERIMENT_GROUPS
        metrics: List of 4 metrics to plot
        output_path: Optional path to save figure
    """
    if experiment_type not in EXPERIMENT_GROUPS:
        raise ValueError(f"Unknown experiment_type: {experiment_type}. "
                        f"Must be one of: {list(EXPERIMENT_GROUPS.keys())}")

    if len(metrics) != 4:
        raise ValueError(f"Must provide exactly 4 metrics, got {len(metrics)}")

    # Get experiment configuration
    exp_config = EXPERIMENT_GROUPS[experiment_type]
    exp_list = exp_config['experiments']
    title = exp_config['title']
    xlabel = exp_config['xlabel']

    # Filter data to this experiment type
    exp_df = df[df['exp_base'].isin(exp_list)].copy()

    if len(exp_df) == 0:
        raise ValueError(f"No data found for experiment_type: {experiment_type}")

    # Filter to only available experiments
    available_exps = [exp for exp in exp_list if exp in exp_df['exp_base'].values]

    if len(available_exps) == 0:
        raise ValueError(f"No matching experiments found in data for {experiment_type}")

    # Create categorical ordering
    exp_df['exp_base'] = pd.Categorical(exp_df['exp_base'],
                                        categories=available_exps,
                                        ordered=True)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Set style
    sns.set_palette("husl")

    # Plot each metric
    summaries = {}
    for idx, metric in enumerate(metrics):
        # Check if metric exists
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data. Skipping.")
            axes[idx].text(0.5, 0.5, f"Metric '{metric}'\nnot found",
                         ha='center', va='center', fontsize=14,
                         transform=axes[idx].transAxes)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue

        # Plot on subplot (show legend only on first subplot)
        summary = plot_single_metric(axes[idx], exp_df, available_exps,
                                     metric, xlabel, show_legend=(idx == 0))
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
    fig.suptitle(f'{title}: Multi-Metric Analysis\n({", ".join([m.upper() for m in metrics])})',
                 fontsize=16, fontweight='bold', y=0.995)

    # Adjust spacing between subplots
    # hspace: vertical spacing between rows (default ~0.2)
    # wspace: horizontal spacing between columns (default ~0.2)
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
        best_exp = summary['mean'].idxmax()
        best_mean = summary.loc[best_exp, 'mean']
        best_std = summary.loc[best_exp, 'std']
        print(f"  → Best: {best_exp} ({best_mean:.4f} ± {best_std:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Create 2x2 grid plot with 4 metrics for sampling ablation experiment type',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot temperature experiments with 4 metrics
  python scripts/ablations/sampling/plot_experiment_type_sampling_ablation.py \\
      --experiment_type temperature \\
      --metrics mce pas der qds

  # Plot policy types with 4 metrics, save to file
  python scripts/ablations/sampling/plot_experiment_type_sampling_ablation.py \\
      --experiment_type policy_type \\
      --metrics mce der pas spread \\
      --output results/ablations/sampling/report/policy_4metrics.pdf

  # Plot sampling strategies with recommended metrics
  python scripts/ablations/sampling/plot_experiment_type_sampling_ablation.py \\
      -e sampling_strategy \\
      -m qds hypervolume spacing mpd \\
      -o results/ablations/sampling/report/sampling_strategy_4metrics.pdf

Available experiment types:
  - temperature: Exploration temperature effects
  - sampling_strategy: Different sampling strategies (greedy, categorical, top_k, top_p)
  - policy_type: On-policy vs off-policy training
  - preference_diversity: Preference distribution effects
  - batch_size: Batch size effects
  - combined: Combined best practices

Common metrics (provide any 4):
  - hypervolume, spacing, spread (traditional MO metrics)
  - tds, mpd (trajectory diversity)
  - mce, pmd (spatial diversity)
  - pas, pfs (preference-aligned metrics)
  - qds, der (composite quality-diversity)
  - fci (flow concentration)
  - rbd (replay buffer diversity)
  - avg_pairwise_distance (spatial coverage)

Recommended combinations by experiment type:
  temperature:          mce pas mpd der
  sampling_strategy:    mce qds hypervolume spacing
  policy_type:          mce der pas spread
  preference_diversity: mce pfs spacing fci
  batch_size:           mce der pas pfs
  combined:             mce der pas qds
        """
    )

    parser.add_argument('--experiment_type', '-e', type=str, required=True,
                       choices=list(EXPERIMENT_GROUPS.keys()),
                       help='Experiment type to plot')

    parser.add_argument('--metrics', '-m', type=str, nargs=4, required=True,
                       help='Four metrics to plot (space-separated)')

    parser.add_argument('--results_csv', '-r', type=str,
                       default='results/ablations/sampling/all_results.csv',
                       help='Path to results CSV file')

    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot (PDF). If not provided, saves to default location.')

    args = parser.parse_args()

    # Load data
    results_path = Path(args.results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"Loading data from: {results_path}")
    df = load_data(results_path)
    print(f"Loaded {len(df)} experiments")

    # Set default output path if not provided
    if args.output is None:
        output_dir = Path('results/ablations/sampling/report')
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_str = '_'.join(args.metrics)
        args.output = str(output_dir / f'{args.experiment_type}_{metrics_str}.pdf')
        print(f"No output path specified, using default: {args.output}")

    # Create plot
    create_four_panel_plot(df, args.experiment_type, args.metrics, args.output)


if __name__ == '__main__':
    main()
