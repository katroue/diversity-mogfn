#!/usr/bin/env python3
"""
Regenerate all_results.csv and summary_by_algorithm.csv from individual algorithm results.

This script combines results from multiple baseline algorithms into comprehensive summary files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def combine_algorithm_results(results_dir: Path, algorithms: list = None):
    """
    Combine individual algorithm result files into all_results.csv

    Args:
        results_dir: Directory containing algorithm result files
        algorithms: List of algorithm names to include (None = auto-detect)
    """
    results_dir = Path(results_dir)

    if algorithms is None:
        # Auto-detect algorithm result files
        result_files = list(results_dir.glob('*_results.csv'))
        algorithms = [f.stem.replace('_results', '') for f in result_files]
        print(f"Auto-detected algorithms: {algorithms}")

    # Combine all algorithm results
    all_dfs = []

    for algorithm in algorithms:
        result_file = results_dir / f"{algorithm}_results.csv"

        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping {algorithm}")
            continue

        print(f"Loading {algorithm} results from {result_file}...")
        df = pd.read_csv(result_file)

        # Ensure 'algorithm' column exists
        if 'algorithm' not in df.columns:
            df['algorithm'] = algorithm

        all_dfs.append(df)
        print(f"  Loaded {len(df)} results for {algorithm}")

    if not all_dfs:
        print("Error: No valid algorithm results found!")
        return None

    # Combine into single DataFrame
    all_results = pd.concat(all_dfs, ignore_index=True)

    # Save combined results
    output_file = results_dir / 'all_results.csv'
    all_results.to_csv(output_file, index=False)
    print(f"\nSaved combined results to {output_file}")
    print(f"Total: {len(all_results)} results across {len(all_dfs)} algorithms")

    return all_results


def generate_summary_by_algorithm(all_results: pd.DataFrame, output_dir: Path):
    """
    Generate summary statistics (mean, std) by algorithm

    Args:
        all_results: DataFrame with all algorithm results
        output_dir: Directory to save summary
    """
    # Group by algorithm and compute statistics
    grouped = all_results.groupby('algorithm')

    # Identify metric columns (exclude metadata)
    exclude_cols = ['algorithm', 'seed', 'num_samples', 'num_samples_for_metrics',
                    'hidden_dim', 'num_layers', 'z_hidden_dim', 'z_num_layers',
                    'pareto_size']

    metric_cols = [col for col in all_results.columns if col not in exclude_cols]

    # Compute mean and std for each metric
    summary_data = {}

    for metric in metric_cols:
        if metric in all_results.columns and pd.api.types.is_numeric_dtype(all_results[metric]):
            summary_data[f'{metric}_mean'] = grouped[metric].mean()
            summary_data[f'{metric}_std'] = grouped[metric].std()
            summary_data[f'{metric}_count'] = grouped[metric].count()

    summary = pd.DataFrame(summary_data)
    summary = summary.reset_index()

    # Reorder columns: algorithm, then metrics alphabetically
    metric_names = sorted(set(col.rsplit('_', 1)[0] for col in summary.columns if col != 'algorithm'))

    ordered_cols = ['algorithm']
    for metric in metric_names:
        if f'{metric}_mean' in summary.columns:
            ordered_cols.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_count'])

    summary = summary[ordered_cols]

    # Save summary
    output_file = output_dir / 'summary_by_algorithm.csv'
    summary.to_csv(output_file, index=False)
    print(f"\nSaved summary statistics to {output_file}")
    print(f"Algorithms: {summary['algorithm'].tolist()}")

    return summary


def print_summary_table(summary: pd.DataFrame):
    """Print a nice summary table of key metrics"""

    print("\n" + "="*100)
    print("SUMMARY BY ALGORITHM - Traditional & Spatial Metrics (All Algorithms)")
    print("="*100)

    # Traditional and spatial metrics (applicable to all algorithms)
    common_metrics = ['hypervolume', 'mce', 'pmd', 'qds', 'training_time']

    print(f"{'Algorithm':<15}", end='')
    for metric in common_metrics:
        if f'{metric}_mean' in summary.columns:
            print(f"{metric.upper():<22}", end='')
    print()
    print("-"*100)

    for _, row in summary.iterrows():
        print(f"{row['algorithm']:<15}", end='')
        for metric in common_metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'

            if mean_col in summary.columns:
                mean_val = row[mean_col]
                std_val = row[std_col] if std_col in summary.columns else 0

                if pd.notna(mean_val):
                    print(f"{mean_val:.4f} ± {std_val:.4f}   ", end='')
                else:
                    print(f"{'N/A':<22}", end='')
        print()
    print("="*100)

    # GFlowNet-specific metrics
    gfn_metrics = ['tds', 'mpd', 'rbd', 'fci', 'final_loss']
    gfn_algos = ['mogfn_pc', 'hngfn']

    gfn_summary = summary[summary['algorithm'].isin(gfn_algos)]

    if not gfn_summary.empty and any(f'{m}_mean' in summary.columns for m in gfn_metrics):
        print("\n" + "="*100)
        print("GFlowNet-Specific Metrics (MOGFN-PC & HN-GFN only)")
        print("="*100)

        print(f"{'Algorithm':<15}", end='')
        for metric in gfn_metrics:
            if f'{metric}_mean' in summary.columns:
                print(f"{metric.upper():<22}", end='')
        print()
        print("-"*100)

        for _, row in gfn_summary.iterrows():
            print(f"{row['algorithm']:<15}", end='')
            for metric in gfn_metrics:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'

                if mean_col in summary.columns:
                    mean_val = row[mean_col]
                    std_val = row[std_col] if std_col in summary.columns else 0

                    if pd.notna(mean_val):
                        print(f"{mean_val:.4f} ± {std_val:.4f}   ", end='')
                    else:
                        print(f"{'N/A':<22}", end='')
            print()
        print("="*100)
        print("Note: TDS, MPD, RBD, FCI require trajectory/flow information (GFlowNets only)")
        print("="*100)


def create_readable_summary(summary: pd.DataFrame, output_dir: Path):
    """
    Create a more readable summary CSV with separate sections for common and GFlowNet metrics
    """
    # Define metric categories
    common_metrics = [
        'hypervolume', 'r2_indicator', 'avg_pairwise_distance', 'spacing', 'spread',
        'mce', 'num_modes', 'pmd', 'pfs', 'num_unique_solutions', 'pas',
        'qds', 'der', 'training_time'
    ]

    gfn_metrics = ['tds', 'mpd', 'rbd', 'fci', 'final_loss']

    # Create common metrics summary (all algorithms)
    common_cols = ['algorithm']
    for metric in common_metrics:
        if f'{metric}_mean' in summary.columns:
            common_cols.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_count'])

    common_summary = summary[common_cols] if all(col in summary.columns for col in common_cols[:4]) else summary

    # Save common metrics summary
    common_file = output_dir / 'summary_common_metrics.csv'
    common_summary.to_csv(common_file, index=False)
    print(f"  - {common_file} (metrics applicable to all algorithms)")

    # Create GFlowNet-specific summary
    gfn_algos = ['mogfn_pc', 'hngfn']
    gfn_summary = summary[summary['algorithm'].isin(gfn_algos)]

    if not gfn_summary.empty:
        gfn_cols = ['algorithm']
        for metric in gfn_metrics:
            if f'{metric}_mean' in summary.columns:
                gfn_cols.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_count'])

        if len(gfn_cols) > 1:
            gfn_metrics_summary = gfn_summary[gfn_cols]
            gfn_file = output_dir / 'summary_gflownet_metrics.csv'
            gfn_metrics_summary.to_csv(gfn_file, index=False)
            print(f"  - {gfn_file} (GFlowNet-specific metrics)")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate summary files from individual algorithm results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/baselines/molecules',
        help='Directory containing algorithm result files'
    )
    parser.add_argument(
        '--algorithms',
        type=str,
        default=None,
        help='Comma-separated list of algorithms (default: auto-detect)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist!")
        return

    # Parse algorithms if provided
    algorithms = None
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(',')]

    print(f"Regenerating summaries for {results_dir}")
    print("="*80)

    # Step 1: Combine all algorithm results
    all_results = combine_algorithm_results(results_dir, algorithms)

    if all_results is None:
        return

    # Step 2: Generate summary by algorithm
    summary = generate_summary_by_algorithm(all_results, results_dir)

    # Step 3: Create readable summaries
    print("\nCreating additional summary files:")
    create_readable_summary(summary, results_dir)

    # Step 4: Print summary table
    print_summary_table(summary)

    print(f"\n✓ Successfully regenerated summary files!")
    print(f"  - {results_dir / 'all_results.csv'}")
    print(f"  - {results_dir / 'summary_by_algorithm.csv'} (comprehensive)")
    print(f"  - {results_dir / 'summary_common_metrics.csv'} (all algorithms)")
    print(f"  - {results_dir / 'summary_gflownet_metrics.csv'} (GFlowNet-only)")


if __name__ == '__main__':
    main()
