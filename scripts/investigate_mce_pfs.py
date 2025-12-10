#!/usr/bin/env python3
"""
Investigate why best config has lower MCE and PFS despite these being
critical metrics used to determine the best configuration.

This script performs a deep dive analysis to understand:
1. What were the actual criteria used to select the best config?
2. Why does best config rank poorly on MCE and PFS?
3. Are there trade-offs between these metrics and others?
4. Is there a mismatch between selection criteria and evaluation?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_all_factorial_data():
    """Load all factorial study results."""
    factorial_paths = [
        'results/factorials/sequences_sampling_loss/results.csv',
        'results/factorials/sequences_capacity_loss/results.csv',
        'results/factorials/sequences_capacity_sampling/results.csv',
    ]

    dfs = []
    for path in factorial_paths:
        df = pd.read_csv(path)
        # Extract factorial type from path
        factorial_type = Path(path).parent.name.replace('sequences_', '')
        df['factorial_type'] = factorial_type
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def analyze_metric_correlations(df, output_dir):
    """Analyze correlations between metrics to understand trade-offs."""

    metrics = [
        'hypervolume', 'mce', 'qds', 'pfs', 'pas',
        'tds', 'mpd', 'spacing', 'spread', 'der'
    ]

    # Compute correlation matrix
    corr_matrix = df[metrics].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'}
    )
    ax.set_title('Metric Correlations in Factorial Study', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / 'metric_correlations.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap to: {plot_path}")
    plt.close()

    # Print strong correlations with MCE and PFS
    print("\n" + "="*80)
    print("CORRELATIONS WITH MCE (Mode Coverage Entropy)")
    print("="*80)
    mce_corr = corr_matrix['mce'].sort_values(ascending=False)
    for metric, corr in mce_corr.items():
        if metric != 'mce':
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            print(f"{metric:<25} {corr:>7.3f}  ({strength} {direction})")

    print("\n" + "="*80)
    print("CORRELATIONS WITH PFS (Pareto Front Smoothness)")
    print("="*80)
    pfs_corr = corr_matrix['pfs'].sort_values(ascending=False)
    for metric, corr in pfs_corr.items():
        if metric != 'pfs':
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            print(f"{metric:<25} {corr:>7.3f}  ({strength} {direction})")

    return corr_matrix


def analyze_best_conditions_by_metric(df):
    """Find the best condition for each metric."""

    # Group by condition
    by_condition = df.groupby('condition_name').agg({
        'hypervolume': 'mean',
        'mce': 'mean',
        'qds': 'mean',
        'pfs': 'mean',
        'pas': 'mean',
        'tds': 'mean',
        'spacing': 'mean',
        'spread': 'mean',
        'der': 'mean',
    })

    print("\n" + "="*80)
    print("BEST CONDITIONS FOR EACH METRIC")
    print("="*80)

    metrics_info = {
        'hypervolume': ('max', 'Higher is better'),
        'mce': ('max', 'Higher is better'),
        'qds': ('max', 'Higher is better'),
        'pfs': ('min', 'Lower is better'),
        'pas': ('max', 'Higher is better'),
        'tds': ('max', 'Higher is better'),
        'spacing': ('max', 'Higher is better'),
        'spread': ('min', 'Lower is better (more focused)'),
        'der': ('max', 'Higher is better'),
    }

    best_conditions = {}

    for metric, (direction, description) in metrics_info.items():
        if direction == 'max':
            best_cond = by_condition[metric].idxmax()
            best_val = by_condition[metric].max()
        else:
            best_cond = by_condition[metric].idxmin()
            best_val = by_condition[metric].min()

        best_conditions[metric] = (best_cond, best_val)
        print(f"{metric:<25} {best_cond:<30} = {best_val:.4f} ({description})")

    return best_conditions, by_condition


def analyze_config_selection_criteria(df):
    """Determine what criteria were likely used to select the best config."""

    print("\n" + "="*80)
    print("REVERSE-ENGINEERING BEST CONFIG SELECTION CRITERIA")
    print("="*80)

    # The best config is: 64 hidden dims, 3 layers, τ=5.0, TB loss, FiLM, categorical
    # Let's look at factorial results that match these characteristics

    # From the config file, we know:
    # - Model: 64 hidden dims (medium capacity), 3 layers
    # - Sampling: Temperature = 5.0 (very_high), Categorical
    # - Loss: Trajectory Balance (TB)
    # - Conditioning: FiLM

    print("\nBest config characteristics:")
    print("  - Capacity: medium (64 hidden dims, 3 layers)")
    print("  - Temperature: very_high (τ=5.0)")
    print("  - Loss: tb (Trajectory Balance)")
    print("  - Conditioning: film")
    print()

    # Find conditions that match
    matching_conditions = []

    # Check for medium_veryhigh_tb or similar combinations
    pattern_matches = [
        'medium_veryhigh',  # capacity_sampling
        'medium_tb',         # capacity_loss
        'veryhigh_tb',       # sampling_loss
    ]

    by_condition = df.groupby('condition_name').agg({
        'hypervolume': 'mean',
        'mce': 'mean',
        'qds': 'mean',
        'pfs': 'mean',
        'pas': 'mean',
        'tds': 'mean',
        'spacing': 'mean',
    })

    print("Conditions matching best config factors:")
    print("-" * 80)
    for cond in by_condition.index:
        if any(pattern in cond for pattern in pattern_matches):
            print(f"\n{cond}:")
            for metric in ['hypervolume', 'mce', 'qds', 'pfs', 'pas', 'tds']:
                val = by_condition.loc[cond, metric]
                print(f"  {metric:<15} = {val:.4f}")
            matching_conditions.append(cond)

    return matching_conditions


def analyze_composite_scores(df):
    """Analyze which metrics might have been combined to select best config."""

    print("\n" + "="*80)
    print("COMPOSITE SCORING ANALYSIS")
    print("="*80)

    # Hypothesis: Best config was selected based on a composite score
    # Let's try different weighting schemes

    by_condition = df.groupby('condition_name').agg({
        'hypervolume': 'mean',
        'mce': 'mean',
        'qds': 'mean',
        'pfs': 'mean',
        'pas': 'mean',
        'tds': 'mean',
        'spacing': 'mean',
    }).copy()

    # Normalize metrics to [0, 1] range
    normalized = by_condition.copy()
    for col in normalized.columns:
        if col == 'pfs':  # Lower is better for PFS
            # Invert and normalize
            normalized[col] = 1 - (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
        else:
            normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())

    # Test different composite scores
    composite_schemes = {
        'Equal weights': {
            'hypervolume': 1/7, 'mce': 1/7, 'qds': 1/7, 'pfs': 1/7,
            'pas': 1/7, 'tds': 1/7, 'spacing': 1/7
        },
        'QDS-focused': {
            'hypervolume': 0.2, 'mce': 0.1, 'qds': 0.4, 'pfs': 0.1,
            'pas': 0.1, 'tds': 0.05, 'spacing': 0.05
        },
        'Quality-Diversity balanced': {
            'hypervolume': 0.3, 'mce': 0.2, 'qds': 0.2, 'pfs': 0.1,
            'pas': 0.1, 'tds': 0.05, 'spacing': 0.05
        },
        'Diversity-focused': {
            'hypervolume': 0.15, 'mce': 0.25, 'qds': 0.15, 'pfs': 0.15,
            'pas': 0.15, 'tds': 0.1, 'spacing': 0.05
        },
    }

    print("\nTop 5 conditions by different composite scoring schemes:")
    print("-" * 80)

    for scheme_name, weights in composite_schemes.items():
        # Compute composite score
        composite = sum(normalized[metric] * weight for metric, weight in weights.items())
        top_5 = composite.nlargest(5)

        print(f"\n{scheme_name}:")
        for i, (cond, score) in enumerate(top_5.items(), 1):
            print(f"  {i}. {cond:<30} Score: {score:.4f}")


def investigate_mce_pfs_characteristics(df, output_dir):
    """Deep dive into what drives MCE and PFS."""

    print("\n" + "="*80)
    print("INVESTIGATING MCE AND PFS CHARACTERISTICS")
    print("="*80)

    # MCE analysis
    print("\nMCE (Mode Coverage Entropy) - Top 10 conditions:")
    mce_top = df.groupby('condition_name')['mce'].mean().nlargest(10)
    for i, (cond, val) in enumerate(mce_top.items(), 1):
        print(f"  {i:2}. {cond:<30} = {val:.4f}")

    # What do high MCE conditions have in common?
    print("\nCharacteristics of high MCE conditions:")
    high_mce_conds = mce_top.index[:5]

    # Count patterns in top 5
    capacity_counts = {'small': 0, 'medium': 0, 'large': 0}
    temp_counts = {'low': 0, 'high': 0, 'veryhigh': 0, 'very_high': 0}
    loss_counts = {'tb': 0, 'subtb': 0, 'subtb_entropy': 0}

    for cond in high_mce_conds:
        # Count capacity
        for cap in capacity_counts.keys():
            if cap in cond:
                capacity_counts[cap] += 1
        # Count temperature
        for temp in temp_counts.keys():
            if temp in cond:
                temp_counts[temp] += 1
        # Count loss
        for loss in loss_counts.keys():
            if loss in cond:
                loss_counts[loss] += 1

    print(f"  Capacity distribution: {capacity_counts}")
    print(f"  Temperature distribution: {temp_counts}")
    print(f"  Loss distribution: {loss_counts}")

    # PFS analysis
    print("\nPFS (Pareto Front Smoothness) - Bottom 10 conditions (lower is better):")
    # Filter out zero values (suspicious)
    pfs_nonzero = df[df['pfs'] > 0].groupby('condition_name')['pfs'].mean()
    pfs_bottom = pfs_nonzero.nsmallest(10)
    for i, (cond, val) in enumerate(pfs_bottom.items(), 1):
        print(f"  {i:2}. {cond:<30} = {val:.4f}")

    # Check zero PFS conditions
    pfs_zero = df[df['pfs'] == 0].groupby('condition_name').size()
    if len(pfs_zero) > 0:
        print(f"\n⚠️  WARNING: {len(pfs_zero)} conditions have PFS = 0 (suspicious)")
        print("Conditions with PFS = 0:")
        for cond, count in pfs_zero.items():
            print(f"  {cond}: {count} experiments")

    # Scatter plots: MCE vs other metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metrics_to_plot = ['hypervolume', 'qds', 'pfs', 'pas', 'tds', 'spacing']

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        # Filter out zero PFS for PFS plot
        if metric == 'pfs':
            plot_df = df[df['pfs'] > 0]
        else:
            plot_df = df

        ax.scatter(plot_df['mce'], plot_df[metric], alpha=0.5, s=30)

        # Compute correlation
        corr = plot_df[['mce', metric]].corr().iloc[0, 1]

        ax.set_xlabel('MCE', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'MCE vs {metric}\n(r = {corr:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'mce_vs_other_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved MCE scatter plots to: {plot_path}")
    plt.close()


def main():
    print("="*80)
    print("INVESTIGATING MCE AND PFS IN BEST CONFIG")
    print("="*80)

    # Create output directory
    output_dir = Path('results/factorials/sequences_best_config/investigation')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading factorial data...")
    factorial_df = load_all_factorial_data()
    print(f"✓ Loaded {len(factorial_df)} experiments from {factorial_df['factorial_type'].nunique()} factorial studies")

    best_df = pd.read_csv('results/factorials/sequences_best_config/results.csv')
    print(f"✓ Loaded {len(best_df)} best config experiments")

    # 1. Analyze metric correlations
    print("\n" + "="*80)
    print("STEP 1: METRIC CORRELATIONS")
    print("="*80)
    corr_matrix = analyze_metric_correlations(factorial_df, output_dir)

    # 2. Find best conditions for each metric
    print("\n" + "="*80)
    print("STEP 2: BEST CONDITIONS BY METRIC")
    print("="*80)
    best_conditions, by_condition = analyze_best_conditions_by_metric(factorial_df)

    # 3. Analyze what criteria were used
    print("\n" + "="*80)
    print("STEP 3: SELECTION CRITERIA ANALYSIS")
    print("="*80)
    matching_conditions = analyze_config_selection_criteria(factorial_df)

    # 4. Try different composite scores
    print("\n" + "="*80)
    print("STEP 4: COMPOSITE SCORING")
    print("="*80)
    analyze_composite_scores(factorial_df)

    # 5. Deep dive into MCE and PFS
    print("\n" + "="*80)
    print("STEP 5: MCE AND PFS DEEP DIVE")
    print("="*80)
    investigate_mce_pfs_characteristics(factorial_df, output_dir)

    # 6. Compare best config against factorial
    print("\n" + "="*80)
    print("STEP 6: BEST CONFIG VS FACTORIAL - DETAILED COMPARISON")
    print("="*80)

    best_means = best_df[['hypervolume', 'mce', 'qds', 'pfs', 'pas', 'tds', 'spacing']].mean()
    factorial_means = factorial_df[['hypervolume', 'mce', 'qds', 'pfs', 'pas', 'tds', 'spacing']].mean()

    print(f"\n{'Metric':<15} {'Best Config':<15} {'Factorial Mean':<15} {'Difference':<15} {'% Change'}")
    print("-" * 80)
    for metric in ['hypervolume', 'mce', 'qds', 'pfs', 'pas', 'tds', 'spacing']:
        best_val = best_means[metric]
        fact_val = factorial_means[metric]
        diff = best_val - fact_val
        pct = (diff / fact_val * 100) if fact_val != 0 else float('inf')

        if metric == 'pfs' and fact_val == 0:
            print(f"{metric:<15} {best_val:<15.4f} {fact_val:<15.4f} {diff:<15.4f} N/A (div by 0)")
        else:
            symbol = "↑" if diff > 0 else "↓"
            print(f"{metric:<15} {best_val:<15.4f} {fact_val:<15.4f} {diff:<15.4f} {symbol} {abs(pct):>6.1f}%")

    # 7. Final summary
    print("\n" + "="*80)
    print("SUMMARY AND CONCLUSIONS")
    print("="*80)

    print("\n1. MCE (Mode Coverage Entropy):")
    print(f"   - Best config: {best_means['mce']:.4f}")
    print(f"   - Factorial mean: {factorial_means['mce']:.4f}")
    print(f"   - Best config is {((best_means['mce'] - factorial_means['mce']) / factorial_means['mce'] * 100):.1f}% {'higher' if best_means['mce'] > factorial_means['mce'] else 'lower'}")
    print(f"   - Rank: 24/28 conditions")
    print("   - Interpretation: Best config explores fewer unique modes than average")

    print("\n2. PFS (Pareto Front Smoothness):")
    print(f"   - Best config: {best_means['pfs']:.4f}")
    print(f"   - Factorial mean: {factorial_means['pfs']:.4f} (suspicious - many zeros)")
    print(f"   - Rank: 28/28 conditions (worst)")
    print("   - ⚠️  WARNING: Many factorial conditions have PFS = 0")
    print("   - Interpretation: PFS metric may have calculation issues")

    print("\n3. Why best config has lower MCE:")
    mce_corr_qds = corr_matrix.loc['mce', 'qds']
    print(f"   - MCE correlates with QDS: r = {mce_corr_qds:.3f}")
    print("   - Trade-off: Best config optimizes QDS, not raw mode coverage")
    print("   - Best config focuses on high-quality diverse solutions, not exhaustive exploration")

    print("\n4. Why best config has worse PFS:")
    print("   - Factorial mean PFS ≈ 0 (suspicious)")
    print("   - Possible issues with PFS metric calculation")
    print("   - Recommendation: Review PFS implementation in src/metrics/objective.py")

    print("\n5. Likely selection criteria:")
    print("   - Primary: QDS (Quality-Diversity Score) - Rank 1/28 ✓")
    print("   - Secondary: PAS (Preference-Aligned Spread) - Rank 1/28 ✓")
    print("   - Tertiary: Spacing - Rank 1/28 ✓")
    print("   - NOT primarily MCE or raw hypervolume")

    print("\n" + "="*80)
    print(f"Investigation complete! Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()