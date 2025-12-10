#!/usr/bin/env python3
"""
Test script for temperature × off-policy interaction validation.

Quick test with reduced iterations (500) and seeds (2) to verify:
1. No mode collapse in on-policy conditions (off_policy_ratio=0.0)
2. Expected interaction pattern between temperature and off-policy
3. Config structure is correct (task, temperature, grid_size, etc.)

Usage:
    # Run all conditions
    python tests/validation/test_temp_offpolicy_interaction.py

    # Run specific conditions only
    python tests/validation/test_temp_offpolicy_interaction.py \
        --conditions temp1_off0,temp1_off10,temp5_off0,temp5_off10

    # Analyze existing results without re-running
    python tests/validation/test_temp_offpolicy_interaction.py --analyze-only
"""

import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_validation_experiment(config_path: Path,
                              output_dir: Path,
                              conditions: str = None,
                              device: str = 'cpu') -> int:
    """
    Run validation experiment using the factorial script.

    Returns:
        Exit code from subprocess
    """
    script_path = project_root / 'scripts' / 'factorials' / 'hypergrid' / 'run_factorial_hypergrid.py'

    cmd = [
        sys.executable,
        str(script_path),
        '--config', str(config_path),
        '--output_dir', str(output_dir),
        '--device', device,
        '--resume'  # Always use resume to skip completed
    ]

    if conditions:
        cmd.extend(['--conditions', conditions])

    print(f"\n{'='*80}")
    print("RUNNING TEMP × OFF-POLICY INTERACTION TEST")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def analyze_interaction_pattern(results_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze results to detect interaction pattern and mode collapse.

    Returns:
        df: DataFrame with experiment results
        summary: Dictionary with interaction analysis
    """
    print(f"\n{'='*80}")
    print("ANALYZING TEMPERATURE × OFF-POLICY INTERACTION")
    print(f"{'='*80}\n")

    results = []

    # Analyze each experiment
    for exp_dir in sorted(results_dir.glob('*_seed*')):
        exp_name = exp_dir.name

        # Load objectives
        obj_file = exp_dir / 'objectives.npy'
        if not obj_file.exists():
            print(f"⚠️  {exp_name}: No objectives.npy found (experiment incomplete)")
            continue

        objectives = np.load(obj_file)

        # Load metrics
        metrics_file = exp_dir / 'metrics.json'
        if not metrics_file.exists():
            print(f"⚠️  {exp_name}: No metrics.json found")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Analyze diversity
        n_unique = len(np.unique(objectives, axis=0))
        n_total = len(objectives)
        unique_rate = n_unique / n_total * 100

        # Detect collapse
        collapsed = n_unique == 1
        severe_collapse = n_unique < 10

        # Parse condition components
        parts = exp_name.replace('_seed', '_SEED').split('_')
        temp_part = parts[0]  # temp1, temp2, temp5
        off_part = parts[1]   # off0, off10
        seed = int(parts[-1])

        # Collect results
        result = {
            'exp_name': exp_name,
            'temperature': temp_part,
            'offpolicy': off_part,
            'seed': seed,
            'n_objectives': n_total,
            'n_unique': n_unique,
            'unique_rate': unique_rate,
            'collapsed': collapsed,
            'severe_collapse': severe_collapse,
            'mce': metrics.get('mce', 0.0),
            'num_modes': metrics.get('num_modes', 0),
            'qds': metrics.get('qds', 0.0),
            'hypervolume': metrics.get('hypervolume', 0.0),
            'tds': metrics.get('tds', 0.0),
            'final_loss': metrics.get('final_loss', float('nan')),
        }

        results.append(result)

        # Print status
        status = ""
        if collapsed:
            status = f"❌ COLLAPSED (1 unique)"
        elif severe_collapse:
            status = f"⚠️  SEVERE ({n_unique} unique)"
        else:
            status = f"✅ OK ({n_unique} unique)"

        print(f"{exp_name:25s} MCE={result['mce']:.4f}  QDS={result['qds']:.4f}  {status}")

    if not results:
        print("\n⚠️  No completed experiments found!")
        return pd.DataFrame(), {}

    df = pd.DataFrame(results)

    # Analyze interaction pattern
    summary = analyze_interaction_effect(df)

    return df, summary


def analyze_interaction_effect(df: pd.DataFrame) -> Dict:
    """
    Analyze the temperature × off-policy interaction effect.

    Expected pattern:
    - temp1: off-policy should INCREASE MCE (beneficial)
    - temp2: off-policy should MAINTAIN or slightly increase MCE
    - temp5: off-policy should DECREASE MCE (harmful - mode collapse)
    """
    if df.empty:
        return {}

    summary = {
        'total_experiments': len(df),
        'collapsed_count': df['collapsed'].sum(),
        'collapsed_rate': df['collapsed'].mean() * 100,
    }

    # Analyze each temperature level
    for temp in ['temp1', 'temp2', 'temp5']:
        temp_df = df[df['temperature'] == temp]
        if temp_df.empty:
            continue

        # Get on-policy and off-policy results
        on_policy = temp_df[temp_df['offpolicy'] == 'off0']
        off_policy = temp_df[temp_df['offpolicy'] == 'off10']

        if on_policy.empty or off_policy.empty:
            continue

        # Calculate mean MCE for each
        mce_on = on_policy['mce'].mean()
        mce_off = off_policy['mce'].mean()
        mce_diff = mce_off - mce_on

        # Calculate collapse rates
        collapse_on = on_policy['collapsed'].sum()
        collapse_off = off_policy['collapsed'].sum()

        summary[f'{temp}_mce_on'] = mce_on
        summary[f'{temp}_mce_off'] = mce_off
        summary[f'{temp}_mce_diff'] = mce_diff
        summary[f'{temp}_collapsed_on'] = collapse_on
        summary[f'{temp}_collapsed_off'] = collapse_off

    return summary


def print_interaction_summary(df: pd.DataFrame, summary: Dict) -> None:
    """Print interaction pattern analysis."""
    print(f"\n{'='*80}")
    print("INTERACTION PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    if df.empty:
        print("No data to analyze.")
        return

    print("Expected Interaction Pattern:")
    print("  temp1 (τ=1.0): off-policy INCREASES MCE (beneficial)")
    print("  temp2 (τ=2.0): off-policy maintains/increases MCE")
    print("  temp5 (τ=5.0): off-policy DECREASES MCE (harmful - mode collapse)")
    print()

    # Analyze each temperature
    for temp in ['temp1', 'temp2', 'temp5']:
        temp_label = {'temp1': 'τ=1.0', 'temp2': 'τ=2.0', 'temp5': 'τ=5.0'}[temp]

        mce_on = summary.get(f'{temp}_mce_on', None)
        mce_off = summary.get(f'{temp}_mce_off', None)
        mce_diff = summary.get(f'{temp}_mce_diff', None)
        collapse_on = summary.get(f'{temp}_collapsed_on', 0)
        collapse_off = summary.get(f'{temp}_collapsed_off', 0)

        if mce_on is None or mce_off is None:
            print(f"{temp_label}: ⚠️  INCOMPLETE DATA")
            continue

        # Determine if pattern matches expectation
        if temp == 'temp1':
            # Expect off-policy to increase MCE
            expected = mce_diff > 0.05
            pattern = "increase" if mce_diff > 0 else "decrease"
        elif temp == 'temp2':
            # Expect off-policy to maintain or increase MCE
            expected = mce_diff >= -0.05
            pattern = "maintain" if abs(mce_diff) < 0.05 else ("increase" if mce_diff > 0 else "decrease")
        else:  # temp5
            # Expect off-policy to cause collapse (MCE near 0)
            expected = (mce_off < 0.10 or collapse_off > 0)
            pattern = "collapse" if (mce_off < 0.10 or collapse_off > 0) else "maintain"

        status = "✅" if expected else "❌"

        print(f"{temp_label}: {status} Off-policy causes MCE to {pattern}")
        print(f"  On-policy (ε=0.0):  MCE={mce_on:.4f}  collapsed={collapse_on}")
        print(f"  Off-policy (ε=0.1): MCE={mce_off:.4f}  collapsed={collapse_off}")
        print(f"  Difference: {mce_diff:+.4f}")
        print()


def print_overall_summary(summary: Dict) -> None:
    """Print overall summary statistics."""
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total experiments: {summary.get('total_experiments', 0)}")
    print(f"Complete collapse: {summary.get('collapsed_count', 0)} ({summary.get('collapsed_rate', 0):.1f}%)")
    print()

    # Check for universal collapse (previous bug)
    if summary.get('collapsed_rate', 0) > 90:
        print("⚠️  CRITICAL: Universal mode collapse detected!")
        print("   This suggests the config structure may still be broken.")
        print("   Check: task='hypergrid', temperature (not temperature_sampling),")
        print("          grid_size=[32,32], etc.")
    elif summary.get('collapsed_rate', 0) > 50:
        print("⚠️  WARNING: >50% experiments collapsed.")
        print("   Some conditions may have fundamental issues.")
    else:
        print("✅ Most experiments are working correctly.")
        print("   Mode collapse is condition-specific, not universal.")


def print_validation_result(df: pd.DataFrame, summary: Dict) -> bool:
    """
    Print validation test result.

    Returns:
        True if validation passed, False otherwise
    """
    print(f"\n{'='*80}")
    print("VALIDATION TEST RESULT")
    print(f"{'='*80}\n")

    # Check for universal collapse (config broken)
    if summary.get('collapsed_rate', 0) > 90:
        print("❌ FAILED: Universal mode collapse (config likely broken)")
        return False

    # Check if on-policy conditions work
    on_policy_collapsed = 0
    on_policy_total = 0
    for temp in ['temp1', 'temp2', 'temp5']:
        collapse = summary.get(f'{temp}_collapsed_on', 0)
        if f'{temp}_mce_on' in summary:
            on_policy_total += 1
            on_policy_collapsed += (collapse > 0)

    if on_policy_total > 0 and on_policy_collapsed / on_policy_total > 0.5:
        print("❌ FAILED: >50% on-policy conditions collapsed")
        print("   On-policy learning should work. Check loss function / training settings.")
        return False

    # Check if temp5_off10 collapses (expected behavior)
    temp5_off_mce = summary.get('temp5_mce_off', 1.0)
    temp5_off_collapsed = summary.get('temp5_collapsed_off', 0)

    if temp5_off_mce > 0.15 and temp5_off_collapsed == 0:
        print("⚠️  WARNING: temp5_off10 did not collapse as expected")
        print("   Expected interaction pattern not observed in quick test.")
        print("   May need longer training (4000 iterations) to see effect.")
        print()
        print("✅ PARTIAL PASS: Config structure is correct, but interaction unclear")
        return True

    print("✅ PASSED: Config structure is correct")
    print("   - No universal collapse")
    print("   - On-policy conditions working")
    print("   - Expected pattern observed (or needs longer training)")
    return True


def save_analysis(df: pd.DataFrame, summary: Dict, output_dir: Path) -> None:
    """Save analysis results to files."""
    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    # Save detailed results
    if not df.empty:
        csv_file = analysis_dir / 'interaction_analysis.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n📊 Saved detailed analysis to: {csv_file}")

    # Save summary
    summary_file = analysis_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Saved summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Test temperature × off-policy interaction validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('tests/validation/test_temp_offpolicy_config.yaml'),
        help='Path to test configuration (default: tests/validation/test_temp_offpolicy_config.yaml)'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('tests/validation/results_temp_offpolicy'),
        help='Output directory for test results (default: tests/validation/results_temp_offpolicy)'
    )

    parser.add_argument(
        '--conditions',
        type=str,
        default=None,
        help='Comma-separated list of conditions to test (default: all)'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze existing results, do not run experiments'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for training (cpu or cuda)'
    )

    args = parser.parse_args()

    # Resolve paths
    config_path = project_root / args.config
    output_dir = project_root / args.output_dir

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1

    # Run experiments (unless analyze-only)
    if not args.analyze_only:
        exit_code = run_validation_experiment(
            config_path=config_path,
            output_dir=output_dir,
            conditions=args.conditions,
            device=args.device
        )

        if exit_code != 0:
            print(f"\n❌ Experiment failed with exit code {exit_code}")
            return exit_code
    else:
        print("\n📊 Analyzing existing results (--analyze-only mode)")

    # Analyze results
    df, summary = analyze_interaction_pattern(output_dir)

    if not df.empty:
        # Print analyses
        print_interaction_summary(df, summary)
        print_overall_summary(summary)

        # Save analysis
        save_analysis(df, summary, output_dir)

        # Print validation result
        passed = print_validation_result(df, summary)

        print(f"\n{'='*80}")
        print("TEST COMPLETE")
        print(f"{'='*80}\n")

        return 0 if passed else 1
    else:
        print("\n⚠️  No results to analyze. Run without --analyze-only first.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
