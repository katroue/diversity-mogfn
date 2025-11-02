#!/usr/bin/env python3
"""
Summarize results from a loss ablation group.

This script creates a CSV summary report for a specific loss ablation group,
aggregating results across seeds and ranking configurations by key metrics.

Usage:
    # Summarize a specific group
    python scripts/ablations/summarize_loss_group.py --group base_loss_comparison

    # Summarize all groups
    python scripts/ablations/summarize_loss_group.py --all

    # Specify custom input/output directories
    python scripts/ablations/summarize_loss_group.py \
        --group entropy_regularization \
        --results_dir results/ablations/loss \
        --output_dir results/ablations/loss/summaries

Example Output:
    results/ablations/loss/base_loss_comparison/
        ├── results.csv                    # Raw results (all seeds)
        └── summary.csv                    # Aggregated summary (mean ± std)
"""

import sys
import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Key metrics for ranking (higher is better for all)
KEY_METRICS = [
    'hypervolume',              # Quality (Pareto front coverage)
    'mode_coverage_entropy',    # Diversity (mode coverage)
    'apd',# Spread (minimum distance between solutions)
    'spacing',# Preference coverage
    'r2_indicator',  # Combined quality-diversity
]

# Metric aliases (for backward compatibility)
METRIC_ALIASES = {
    'mce': 'mode_coverage_entropy',
    'pmd': 'pairwise_minimum_distance',
    'apd': 'average_pairwise_distance',
    'qds': 'quality_diversity_score',
}


def load_loss_ablation_config() -> dict:
    """Load the loss ablation configuration."""
    config_path = project_root / 'configs' / 'ablations' / 'loss_ablation.yaml'
    if not config_path.exists():
        # Try alternative location
        config_path = project_root / 'configs' / 'ablations' / 'loss_ablation_final.yaml'

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_available_groups(config: dict) -> List[str]:
    """Get list of all experiment groups."""
    experiments = config.get('experiments', [])
    return [exp['group'] for exp in experiments]


def load_group_results(group_name: str, results_dir: Path) -> Optional[pd.DataFrame]:
    """Load results CSV for a specific group."""
    group_dir = results_dir / group_name
    results_csv = group_dir / 'results.csv'

    if not results_csv.exists():
        print(f"⚠️  No results found for group '{group_name}' at {results_csv}")
        return None

    df = pd.read_csv(results_csv)
    print(f"✓ Loaded {len(df)} experiments for group '{group_name}'")
    return df


def standardize_metric_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize metric column names using aliases."""
    df = df.copy()

    # Apply aliases
    for alias, full_name in METRIC_ALIASES.items():
        if alias in df.columns and full_name not in df.columns:
            df[full_name] = df[alias]

    return df


def aggregate_by_configuration(df: pd.DataFrame, vary_factor: str, group_name: str) -> pd.DataFrame:
    """
    Aggregate results by configuration (across seeds).

    Args:
        df: DataFrame with raw results
        vary_factor: Name of the factor that varies (e.g., 'base_loss', 'regularization')
        group_name: Name of the experiment group (e.g., 'base_loss_comparison')

    Returns:
        Aggregated DataFrame with mean and std for each metric
    """
    # Standardize metric names
    df = standardize_metric_names(df)

    # Determine grouping column
    if vary_factor in df.columns:
        group_col = vary_factor
    elif 'exp_name' in df.columns:
        # Extract configuration from exp_name by removing seed suffix
        # e.g., "base_loss_comparison_trajectory_balance_seed42" -> "base_loss_comparison_trajectory_balance"
        group_col = 'config'
        df[group_col] = df['exp_name'].str.replace(r'_seed\d+$', '', regex=True)

        # Remove group name prefix to get just the configuration
        # e.g., "base_loss_comparison_trajectory_balance" -> "trajectory_balance"
        # Handle both patterns:
        #   1. "base_loss_comparison_trajectory_balance"
        #   2. "trajectory_balance" (already short form)
        group_prefix = group_name + '_'
        df[group_col] = df[group_col].str.replace(f'^{group_prefix}', '', regex=True)

        # Also try removing common prefixes like "base_loss_comparison_"
        # This handles legacy naming conventions
        common_prefixes = [
            'base_loss_comparison_',
            'entropy_regularization_',
            'kl_regularization_',
            'subtb_entropy_sweep_',
            'loss_modifications_'
        ]
        for prefix in common_prefixes:
            if prefix != group_prefix:
                df[group_col] = df[group_col].str.replace(f'^{prefix}', '', regex=True)

    else:
        print("⚠️  Could not determine grouping column")
        return df

    # Get all numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude seed and other non-metric columns
    exclude_cols = ['seed', 'num_parameters', 'training_time', 'final_loss']
    metric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Aggregate: compute mean and std across seeds
    agg_dict = {}
    for col in metric_cols:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std'] = (col, 'std')
    agg_dict['num_seeds'] = (group_col, 'count')

    summary = df.groupby(group_col).agg(**agg_dict).reset_index()

    # Rename group column
    summary.rename(columns={group_col: 'configuration'}, inplace=True)

    return summary


def rank_configurations(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add ranking columns for key metrics.

    Ranks are 1-indexed, with 1 being the best.
    """
    summary = summary.copy()

    for metric in KEY_METRICS:
        mean_col = f'{metric}_mean'
        if mean_col in summary.columns:
            # Rank in descending order (higher is better)
            summary[f'{metric}_rank'] = summary[mean_col].rank(ascending=False, method='min')

    # Compute average rank across all metrics
    rank_cols = [f'{metric}_rank' for metric in KEY_METRICS
                 if f'{metric}_rank' in summary.columns]

    if rank_cols:
        summary['avg_rank'] = summary[rank_cols].mean(axis=1)
        summary['rank_std'] = summary[rank_cols].std(axis=1)

    return summary


def create_summary_report(group_name: str,
                         results_dir: Path,
                         output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Create a summary report for a loss ablation group.

    Args:
        group_name: Name of the experiment group
        results_dir: Directory containing loss ablation results
        output_dir: Directory to save summary (defaults to group directory)

    Returns:
        Path to summary CSV file, or None if failed
    """
    print("\n" + "="*80)
    print(f"SUMMARIZING LOSS ABLATION GROUP: {group_name}")
    print("="*80)

    # Load raw results
    df = load_group_results(group_name, results_dir)
    if df is None:
        return None

    # Load config to get vary factor
    config = load_loss_ablation_config()
    experiments = config.get('experiments', [])
    group_config = next((exp for exp in experiments if exp['group'] == group_name), None)

    if group_config is None:
        print(f"⚠️  Group '{group_name}' not found in config")
        vary_factor = 'configuration'
    else:
        vary_factors = list(group_config.get('vary', {}).keys())
        vary_factor = vary_factors[0] if vary_factors else 'configuration'
        print(f"Varying factor: {vary_factor}")

    # Aggregate by configuration
    summary = aggregate_by_configuration(df, vary_factor, group_name)
    print(f"✓ Aggregated {len(summary)} unique configurations")

    # Rank configurations
    summary = rank_configurations(summary)

    # Sort by average rank (best first)
    if 'avg_rank' in summary.columns:
        summary = summary.sort_values('avg_rank')

    # Determine output path
    if output_dir is None:
        output_dir = results_dir / group_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'summary.csv'
    summary.to_csv(output_path, index=False)
    print(f"✓ Summary saved to: {output_path}")

    # Print top configurations
    print("\n" + "-"*80)
    print("TOP CONFIGURATIONS")
    print("-"*80)

    display_cols = ['configuration']
    for metric in KEY_METRICS:
        if f'{metric}_mean' in summary.columns:
            display_cols.extend([f'{metric}_mean', f'{metric}_rank'])
    if 'avg_rank' in summary.columns:
        display_cols.append('avg_rank')

    # Show top 3 configurations
    top_n = min(3, len(summary))
    top_configs = summary[display_cols].head(top_n)

    print(f"\nTop {top_n} configurations by average rank:")
    print(top_configs.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Show winner for each metric
    print("\n" + "-"*80)
    print("BEST CONFIGURATION PER METRIC")
    print("-"*80)

    for metric in KEY_METRICS:
        mean_col = f'{metric}_mean'
        if mean_col in summary.columns:
            best_idx = summary[mean_col].idxmax()
            best_config = summary.loc[best_idx, 'configuration']
            best_value = summary.loc[best_idx, mean_col]
            best_std = summary.loc[best_idx, f'{metric}_std']
            print(f"{metric:30s}: {best_config:30s} ({best_value:.4f} ± {best_std:.4f})")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Summarize loss ablation group results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize a specific group
  python scripts/ablations/summarize_loss_group.py --group base_loss_comparison

  # Summarize all groups
  python scripts/ablations/summarize_loss_group.py --all

  # List available groups
  python scripts/ablations/summarize_loss_group.py --list
        """
    )

    parser.add_argument('--group', type=str,
                       help='Name of the experiment group to summarize')
    parser.add_argument('--all', action='store_true',
                       help='Summarize all groups')
    parser.add_argument('--list', action='store_true',
                       help='List available groups and exit')
    parser.add_argument('--results_dir', type=Path,
                       default=project_root / 'results' / 'ablations' / 'loss',
                       help='Directory containing loss ablation results')
    parser.add_argument('--output_dir', type=Path,
                       help='Directory to save summaries (default: results_dir/group_name)')

    args = parser.parse_args()

    # Load config
    config = load_loss_ablation_config()
    available_groups = get_available_groups(config)

    # List groups if requested
    if args.list:
        print("\n" + "="*80)
        print("AVAILABLE LOSS ABLATION GROUPS")
        print("="*80)
        for i, group in enumerate(available_groups, 1):
            print(f"{i}. {group}")
        print("="*80 + "\n")
        return

    # Validate arguments
    if not args.all and not args.group:
        parser.error("Either --group or --all must be specified")

    if args.group and args.group not in available_groups:
        print(f"⚠️  Unknown group: {args.group}")
        print(f"Available groups: {', '.join(available_groups)}")
        print("Use --list to see all available groups")
        sys.exit(1)

    # Process groups
    groups_to_process = available_groups if args.all else [args.group]

    print("\n" + "="*80)
    print("LOSS ABLATION GROUP SUMMARY")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Groups to process: {len(groups_to_process)}")
    print("="*80)

    success_count = 0
    for group in groups_to_process:
        result = create_summary_report(
            group_name=group,
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        if result is not None:
            success_count += 1

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY COMPLETE")
    print("="*80)
    print(f"Successfully summarized: {success_count}/{len(groups_to_process)} groups")

    if success_count < len(groups_to_process):
        print(f"Failed: {len(groups_to_process) - success_count} groups")
        print("⚠️  Some groups may not have results yet")


if __name__ == '__main__':
    main()
