#!/usr/bin/env python3
"""
Analyze Temperature × Off-Policy Interaction Validation Results

This script analyzes the results of the validation experiment to test whether
off-policy exploration has a non-linear interaction with temperature sampling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def load_results(results_dir: str) -> pd.DataFrame:
    """Load validation results"""
    results_path = Path(results_dir) / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    df = pd.read_csv(results_path)

    # Extract temperature and off-policy from condition names
    df['temperature'] = df['condition_name'].str.extract(r'temp(\d+)')[0].astype(float)
    df['off_policy'] = df['condition_name'].str.extract(r'off(\d+)')[0].astype(float) / 100

    return df

def print_summary(df: pd.DataFrame):
    """Print summary statistics"""
    print("=" * 80)
    print("TEMPERATURE × OFF-POLICY INTERACTION: Results Summary")
    print("=" * 80)
    print()

    # Group by temperature and off_policy
    summary = df.groupby(['temperature', 'off_policy']).agg({
        'mce': ['mean', 'std', 'count'],
        'num_modes': ['mean', 'std'],
        'qds': ['mean', 'std']
    }).round(4)

    print(summary)
    print()

def test_hypotheses(df: pd.DataFrame):
    """Test validation hypotheses"""
    print("=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)
    print()

    # H1: Off-policy improves MCE at temp=1.0
    temp1_off0 = df[(df['temperature'] == 1.0) & (df['off_policy'] == 0.0)]['mce']
    temp1_off10 = df[(df['temperature'] == 1.0) & (df['off_policy'] == 0.1)]['mce']

    if len(temp1_off0) > 0 and len(temp1_off10) > 0:
        t_stat, p_val = stats.ttest_ind(temp1_off10, temp1_off0)
        improvement = temp1_off10.mean() - temp1_off0.mean()

        print(f"H1: Off-policy improves MCE at τ=1.0")
        print(f"  MCE(off=0.0): {temp1_off0.mean():.4f} ± {temp1_off0.std():.4f}")
        print(f"  MCE(off=0.1): {temp1_off10.mean():.4f} ± {temp1_off10.std():.4f}")
        print(f"  Improvement: {improvement:.4f}")
        print(f"  t-test: t={t_stat:.2f}, p={p_val:.4f}")
        print(f"  Result: {'✓ CONFIRMED' if p_val < 0.05 and improvement > 0 else '✗ NOT CONFIRMED'}")
        print()

    # H3: Off-policy causes collapse at temp=5.0
    temp5_off0 = df[(df['temperature'] == 5.0) & (df['off_policy'] == 0.0)]['mce']
    temp5_off10 = df[(df['temperature'] == 5.0) & (df['off_policy'] == 0.1)]['mce']

    if len(temp5_off0) > 0 and len(temp5_off10) > 0:
        t_stat, p_val = stats.ttest_ind(temp5_off0, temp5_off10)
        collapse = temp5_off0.mean() - temp5_off10.mean()

        print(f"H3: Off-policy causes mode collapse at τ=5.0")
        print(f"  MCE(off=0.0): {temp5_off0.mean():.4f} ± {temp5_off0.std():.4f}")
        print(f"  MCE(off=0.1): {temp5_off10.mean():.4f} ± {temp5_off10.std():.4f}")
        print(f"  Decrease: {collapse:.4f}")
        print(f"  t-test: t={t_stat:.2f}, p={p_val:.4f}")

        # Check if collapse occurred
        collapsed = temp5_off10.mean() < 0.05
        print(f"  Mode collapse (MCE < 0.05): {collapsed}")
        print(f"  Result: {'✓ CONFIRMED' if collapsed else '✗ NOT CONFIRMED'}")
        print()

    # H4: Non-linear interaction
    print(f"H4: Temperature × Off-Policy interaction is non-linear")
    print(f"  Pattern: Effect of off-policy reverses at high temperature")
    if len(temp1_off0) > 0 and len(temp5_off0) > 0:
        effect_at_temp1 = temp1_off10.mean() - temp1_off0.mean() if len(temp1_off10) > 0 else 0
        effect_at_temp5 = temp5_off10.mean() - temp5_off0.mean() if len(temp5_off10) > 0 else 0

        print(f"  Effect at τ=1.0: {effect_at_temp1:+.4f} (off-policy {'helps' if effect_at_temp1 > 0 else 'hurts'})")
        print(f"  Effect at τ=5.0: {effect_at_temp5:+.4f} (off-policy {'helps' if effect_at_temp5 > 0 else 'hurts'})")
        print(f"  Sign reversal: {effect_at_temp1 > 0 and effect_at_temp5 < 0}")
        print(f"  Result: {'✓ CONFIRMED (non-linear interaction)' if (effect_at_temp1 > 0 and effect_at_temp5 < 0) else '✗ NOT CONFIRMED'}")
    print()

def plot_interaction(df: pd.DataFrame, output_dir: str):
    """Create interaction plot"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['mce', 'num_modes', 'qds']
    titles = ['Mode Coverage Entropy (MCE)', 'Number of Modes', 'Quality-Diversity Score (QDS)']

    for ax, metric, title in zip(axes, metrics, titles):
        # Compute means and stds
        summary = df.groupby(['temperature', 'off_policy'])[metric].agg(['mean', 'std']).reset_index()

        for off_policy in [0.0, 0.1]:
            data = summary[summary['off_policy'] == off_policy]
            label = f"off-policy={off_policy:.1f}"
            ax.plot(data['temperature'], data['mean'], marker='o', label=label, linewidth=2)
            ax.fill_between(data['temperature'],
                           data['mean'] - data['std'],
                           data['mean'] + data['std'],
                           alpha=0.2)

        ax.set_xlabel('Temperature (τ)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'temp_offpolicy_interaction.png', dpi=300, bbox_inches='tight')
    print(f"Saved interaction plot to: {output_dir}/temp_offpolicy_interaction.png")
    plt.close()

def main():
    """Main analysis"""
    results_dir = "results/validation/temp_offpolicy"

    try:
        # Load results
        df = load_results(results_dir)

        # Print summary
        print_summary(df)

        # Test hypotheses
        test_hypotheses(df)

        # Create plots
        plot_interaction(df, results_dir)

        print("=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        print()
        print("If H3 is confirmed (mode collapse at τ=5.0 with off-policy=0.1):")
        print("  → This validates your factorial study methodology")
        print("  → The 'best_config' collapse was due to parameter interaction")
        print("  → You discovered a novel non-linear interaction")
        print("  → Your corrected configs (off-policy=0.0) should work well")
        print()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print()
        print("Run the validation experiment first:")
        print("  bash scripts/validation/run_temp_offpolicy.sh")

if __name__ == "__main__":
    main()
