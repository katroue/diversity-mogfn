#!/usr/bin/env python3
"""
Compare best config results against factorial study results for DNA sequences task.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Key diversity metrics to analyze
KEY_METRICS = [
    'hypervolume',           # Traditional: Pareto front quality
    'mce',                   # Spatial: Mode Coverage Entropy
    'qds',                   # Composite: Quality-Diversity Score
    'pfs',                   # Objective: Pareto Front Smoothness (lower is better)
    'pas',                   # Objective: Preference-Aligned Spread
    'tds',                   # Trajectory: Trajectory Diversity Score
    'mpd',                   # Trajectory: Multi-Path Diversity
    'spacing',               # Traditional: Solution spacing
    'spread',                # Traditional: Solution spread
    'avg_pairwise_distance', # Distance metric
    'der',                   # Composite: Diversity-Efficiency Ratio
]

def load_and_analyze():
    """Load all results and perform comparison analysis."""

    # Load best config results
    best_config_path = Path('results/factorials/sequences_best_config/results.csv')
    best_df = pd.read_csv(best_config_path)

    # Load factorial study results
    factorial_paths = [
        'results/factorials/sequences_sampling_loss/results.csv',
        'results/factorials/sequences_capacity_loss/results.csv',
        'results/factorials/sequences_capacity_sampling/results.csv',
    ]

    factorial_dfs = []
    for path in factorial_paths:
        df = pd.read_csv(path)
        factorial_dfs.append(df)

    # Combine all factorial results
    factorial_df = pd.concat(factorial_dfs, ignore_index=True)

    print("="*80)
    print("BEST CONFIG vs FACTORIAL STUDY COMPARISON")
    print("DNA Sequences Task Analysis")
    print("="*80)
    print()

    # Best config statistics
    print("BEST CONFIG RESULTS (5 seeds)")
    print("-"*80)
    best_stats = best_df[KEY_METRICS].agg(['mean', 'std', 'min', 'max'])
    print(best_stats.round(4))
    print()

    # Factorial study statistics
    print("FACTORIAL STUDY RESULTS (all conditions)")
    print("-"*80)
    print(f"Total experiments: {len(factorial_df)}")
    print(f"Unique conditions: {factorial_df['condition_name'].nunique()}")
    factorial_stats = factorial_df[KEY_METRICS].agg(['mean', 'std', 'min', 'max'])
    print(factorial_stats.round(4))
    print()

    # Comparison: Best config vs factorial mean
    print("COMPARISON: Best Config Mean vs Factorial Mean")
    print("-"*80)
    comparison = pd.DataFrame({
        'Best Config Mean': best_df[KEY_METRICS].mean(),
        'Factorial Mean': factorial_df[KEY_METRICS].mean(),
        'Difference': best_df[KEY_METRICS].mean() - factorial_df[KEY_METRICS].mean(),
        'Improvement %': ((best_df[KEY_METRICS].mean() - factorial_df[KEY_METRICS].mean()) /
                          factorial_df[KEY_METRICS].mean() * 100)
    })
    print(comparison.round(4))
    print()

    # Statistical significance (t-test)
    from scipy import stats
    print("STATISTICAL SIGNIFICANCE (vs Factorial Mean)")
    print("-"*80)
    print(f"{'Metric':<25} {'Best Mean':<12} {'Fact Mean':<12} {'t-stat':<12} {'p-value':<12} {'Significant'}")
    print("-"*80)

    for metric in KEY_METRICS:
        best_mean = best_df[metric].mean()
        factorial_mean = factorial_df[metric].mean()

        # One-sample t-test: does best config mean differ from factorial mean?
        t_stat, p_val = stats.ttest_1samp(best_df[metric], factorial_mean)

        is_significant = "✓" if p_val < 0.05 else ""
        print(f"{metric:<25} {best_mean:<12.4f} {factorial_mean:<12.4f} {t_stat:<12.4f} {p_val:<12.4f} {is_significant}")

    print()

    # Find best factorial condition for comparison
    print("COMPARISON: Best Config vs Best Factorial Condition")
    print("-"*80)

    factorial_by_condition = factorial_df.groupby('condition_name')[KEY_METRICS].mean()

    # For each metric, find the best factorial condition
    best_factorial_conditions = {}
    for metric in KEY_METRICS:
        # Lower is better for PFS, higher is better for others
        if metric == 'pfs':
            best_cond = factorial_by_condition[metric].idxmin()
        else:
            best_cond = factorial_by_condition[metric].idxmax()
        best_factorial_conditions[metric] = best_cond

    print(f"{'Metric':<25} {'Best Config':<12} {'Best Factorial':<12} {'Condition':<30} {'Better?'}")
    print("-"*80)

    for metric in KEY_METRICS:
        best_config_val = best_df[metric].mean()
        best_factorial_cond = best_factorial_conditions[metric]
        best_factorial_val = factorial_by_condition.loc[best_factorial_cond, metric]

        # Determine if best config is better
        if metric == 'pfs':  # Lower is better
            is_better = "✓" if best_config_val < best_factorial_val else "✗"
        else:  # Higher is better
            is_better = "✓" if best_config_val > best_factorial_val else "✗"

        print(f"{metric:<25} {best_config_val:<12.4f} {best_factorial_val:<12.4f} {best_factorial_cond:<30} {is_better}")

    print()

    # Rank best config among all factorial conditions
    print("RANKING: Best Config vs All Factorial Conditions")
    print("-"*80)

    # Add best config as a condition to factorial stats
    factorial_by_condition_with_best = pd.concat([
        factorial_by_condition,
        pd.DataFrame(best_df[KEY_METRICS].mean()).T.rename(index={0: 'BEST_CONFIG'})
    ])

    for metric in KEY_METRICS:
        if metric == 'pfs':  # Lower is better
            rank = (factorial_by_condition_with_best[metric] <=
                   factorial_by_condition_with_best.loc['BEST_CONFIG', metric]).sum()
        else:  # Higher is better
            rank = (factorial_by_condition_with_best[metric] >=
                   factorial_by_condition_with_best.loc['BEST_CONFIG', metric]).sum()

        total_conditions = len(factorial_by_condition_with_best)
        percentile = (total_conditions - rank + 1) / total_conditions * 100

        print(f"{metric:<25} Rank: {rank}/{total_conditions} (Top {percentile:.1f}%)")

    print()

    # Success criteria evaluation (from config file)
    print("SUCCESS CRITERIA EVALUATION")
    print("-"*80)

    criteria = {
        'mce': (0.30, 'minimum', '>'),
        'hypervolume': (0.55, 'minimum', '>'),
        'qds': (0.60, 'minimum', '>'),
        'pfs': (0.30, 'maximum', '<'),
    }

    print(f"{'Metric':<25} {'Best Mean':<12} {'Criterion':<15} {'Met?'}")
    print("-"*80)

    for metric, (threshold, desc, op) in criteria.items():
        best_mean = best_df[metric].mean()

        if op == '>':
            met = "✓" if best_mean > threshold else "✗"
            criterion_str = f"> {threshold}"
        else:
            met = "✓" if best_mean < threshold else "✗"
            criterion_str = f"< {threshold}"

        print(f"{metric:<25} {best_mean:<12.4f} {criterion_str:<15} {met}")

    print()

    # Coefficient of variation (stability check)
    print("STABILITY ANALYSIS (Coefficient of Variation)")
    print("-"*80)
    print(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'CV':<12} {'Stable?'}")
    print("-"*80)

    for metric in KEY_METRICS:
        mean_val = best_df[metric].mean()
        std_val = best_df[metric].std()
        cv = std_val / mean_val if mean_val != 0 else float('inf')

        # Stable if CV < 0.20 (from config)
        is_stable = "✓" if cv < 0.20 else "✗"

        print(f"{metric:<25} {mean_val:<12.4f} {std_val:<12.4f} {cv:<12.4f} {is_stable}")

    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    # Count how many metrics best config is better than factorial mean
    better_count = 0
    for metric in KEY_METRICS:
        best_val = best_df[metric].mean()
        factorial_val = factorial_df[metric].mean()

        if metric == 'pfs':
            if best_val < factorial_val:
                better_count += 1
        else:
            if best_val > factorial_val:
                better_count += 1

    print(f"Best config outperforms factorial mean on {better_count}/{len(KEY_METRICS)} metrics")

    # Check success criteria
    success_count = sum([
        best_df['mce'].mean() > 0.30,
        best_df['hypervolume'].mean() > 0.55,
        best_df['qds'].mean() > 0.60,
        best_df['pfs'].mean() < 0.30,
    ])

    print(f"Success criteria met: {success_count}/4")

    # Check stability
    stable_count = sum([
        (best_df[metric].std() / best_df[metric].mean() < 0.20)
        if best_df[metric].mean() != 0 else False
        for metric in KEY_METRICS
    ])

    print(f"Stable metrics (CV < 0.20): {stable_count}/{len(KEY_METRICS)}")

    print()
    print("="*80)

if __name__ == '__main__':
    load_and_analyze()