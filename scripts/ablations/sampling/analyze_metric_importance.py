#!/usr/bin/env python3
"""
Analyze which metrics show the most important distinctions between configurations
for each experiment type in the sampling ablation study.

Identifies metrics with:
1. High variance between configurations (separation)
2. Low variance within configurations (consistency across seeds)
3. High signal-to-noise ratio
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Experiment type mappings
EXPERIMENT_GROUPS = {
    'temperature': ['temp_low', 'temp_medium', 'temp_high', 'temp_very_high'],
    'sampling_strategy': ['greedy', 'categorical', 'top_k', 'top_p'],
    'policy_type': ['on_policy_pure', 'off_policy_10', 'off_policy_25', 'off_policy_50'],
    'preference_diversity': ['pref_uniform', 'pref_dirichlet_low', 'pref_dirichlet_medium', 'pref_dirichlet_high'],
    'batch_size': ['batch_32', 'batch_64', 'batch_256', 'batch_512'],
    'combined': ['diverse_sampling', 'quality_sampling']
}

# Metrics to analyze (excluding MCE as requested)
ALL_METRICS = [
    'hypervolume', 'spacing', 'spread', 'avg_pairwise_distance',
    'tds', 'mpd', 'num_modes', 'pmd', 'pfs', 'pas',
    'rbd', 'fci', 'qds', 'der'
]


def calculate_metric_importance(df, exp_list, metric):
    """
    Calculate importance score for a metric based on:
    - Between-group variance (separation between configs)
    - Within-group variance (consistency across seeds)
    - Signal-to-noise ratio
    """
    exp_data = df[df['exp_base'].isin(exp_list)]

    if len(exp_data) == 0:
        return None

    # Calculate between-group variance (variance of means)
    config_means = exp_data.groupby('exp_base')[metric].mean()
    between_var = config_means.var()

    # Calculate within-group variance (average variance within each config)
    config_vars = exp_data.groupby('exp_base')[metric].var()
    within_var = config_vars.mean()

    # Calculate range (max - min of means)
    metric_range = config_means.max() - config_means.min()

    # Signal-to-noise ratio
    if within_var > 0:
        snr = between_var / within_var
    else:
        snr = np.inf if between_var > 0 else 0

    # Coefficient of variation between configs (relative variability)
    mean_of_means = config_means.mean()
    if mean_of_means != 0:
        cv = config_means.std() / abs(mean_of_means)
    else:
        cv = 0

    # Normalized range (range / mean, shows relative difference)
    if mean_of_means != 0:
        normalized_range = metric_range / abs(mean_of_means)
    else:
        normalized_range = 0

    # Composite importance score
    # Higher between_var, lower within_var, higher snr = more important
    importance = snr * normalized_range

    return {
        'metric': metric,
        'between_var': between_var,
        'within_var': within_var,
        'snr': snr,
        'range': metric_range,
        'cv': cv,
        'normalized_range': normalized_range,
        'importance_score': importance,
        'n_configs': len(config_means),
        'mean_value': mean_of_means
    }


def analyze_experiment_type(df, exp_type, exp_list):
    """Analyze all metrics for a given experiment type."""
    print(f"\n{'='*80}")
    print(f"Experiment Type: {exp_type.upper()}")
    print(f"Configurations: {exp_list}")
    print(f"{'='*80}")

    results = []
    for metric in ALL_METRICS:
        if metric not in df.columns:
            continue

        result = calculate_metric_importance(df, exp_list, metric)
        if result is not None:
            results.append(result)

    # Create DataFrame and sort by importance
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('importance_score', ascending=False)

    # Display top metrics
    print("\nTop 10 Most Distinguishing Metrics:")
    print(results_df[['metric', 'importance_score', 'snr', 'normalized_range',
                     'between_var', 'within_var']].head(10).to_string(index=False))

    # Find top 4 metrics (excluding MCE which was already requested)
    top_4 = results_df.head(4)['metric'].tolist()

    print(f"\n🎯 TOP 4 RECOMMENDED METRICS (excluding MCE):")
    for i, metric in enumerate(top_4, 1):
        row = results_df[results_df['metric'] == metric].iloc[0]
        print(f"  {i}. {metric.upper():25s} (score: {row['importance_score']:.2f}, SNR: {row['snr']:.2f})")

    return results_df, top_4


def main():
    # Load data
    results_path = Path('results/ablations/sampling/all_results.csv')

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    print(f"Loading data from: {results_path}")
    df = pd.read_csv(results_path)

    # Extract base experiment name
    df['exp_base'] = df['exp_name'].str.replace(r'_seed\d+$', '', regex=True)

    print(f"Loaded {len(df)} experiments")
    print(f"Unique configurations: {df['exp_base'].nunique()}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    # Analyze each experiment type
    all_recommendations = {}

    for exp_type, exp_list in EXPERIMENT_GROUPS.items():
        results_df, top_4 = analyze_experiment_type(df, exp_type, exp_list)
        all_recommendations[exp_type] = top_4

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: TOP 4 METRICS FOR EACH EXPERIMENT TYPE (excluding MCE)")
    print("="*80)

    for exp_type, metrics in all_recommendations.items():
        print(f"\n{exp_type.upper():20s}: {', '.join([m.upper() for m in metrics])}")

    # Find most commonly recommended metrics across all experiment types
    from collections import Counter
    all_top_metrics = [m for metrics in all_recommendations.values() for m in metrics]
    metric_counts = Counter(all_top_metrics)

    print("\n" + "="*80)
    print("MOST FREQUENTLY RECOMMENDED METRICS ACROSS ALL EXPERIMENT TYPES:")
    print("="*80)
    for metric, count in metric_counts.most_common(10):
        print(f"  {metric.upper():25s}: appears in {count}/6 experiment types")

    # Suggest a universal set of 4 metrics (excluding MCE)
    print("\n" + "="*80)
    print("🌟 UNIVERSAL RECOMMENDATION (4 metrics that work well across all experiments):")
    print("="*80)
    universal_metrics = [m for m, _ in metric_counts.most_common(4)]
    for i, metric in enumerate(universal_metrics, 1):
        count = metric_counts[metric]
        print(f"  {i}. {metric.upper():25s} (important in {count}/6 experiment types)")


if __name__ == '__main__':
    main()
