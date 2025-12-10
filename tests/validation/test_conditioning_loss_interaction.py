#!/usr/bin/env python3
"""
Test script for conditioning × loss interaction validation.

Quick test with reduced iterations (500) and seeds (2) to verify:
1. No mode collapse in any conditions
2. Both concat and FiLM conditioning work correctly
3. Config structure is correct (task, temperature, grid_size, etc.)

Usage:
    # Run all conditions
    python tests/validation/test_conditioning_loss_interaction.py

    # Run specific conditions only
    python tests/validation/test_conditioning_loss_interaction.py \
        --conditions concat_tb,concat_subtb_entropy,film_tb,film_subtb_entropy

    # Analyze existing results without re-running
    python tests/validation/test_conditioning_loss_interaction.py --analyze-only
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
    print("RUNNING CONDITIONING × LOSS INTERACTION TEST")
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
    print("ANALYZING CONDITIONING × LOSS INTERACTION")
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
        # concat_tb_seed42 -> ['concat', 'tb', 'SEED42']
        # film_subtb_entropy_seed42 -> ['film', 'subtb', 'entropy', 'SEED42']

        # Find seed index (last part with SEED)
        seed_idx = next(i for i, p in enumerate(parts) if 'SEED' in p)
        seed = int(parts[seed_idx].replace('SEED', ''))

        # Conditioning is first part
        conditioning = parts[0]

        # Loss is everything between conditioning and seed
        loss_parts = parts[1:seed_idx]
        loss = '_'.join(loss_parts)

        # Collect results
        result = {
            'exp_name': exp_name,
            'conditioning': conditioning,
            'loss': loss,
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
            status = f"❌ COLLAPSED"
        elif severe_collapse:
            status = f"⚠️  SEVERE ({n_unique} unique)"
        else:
            status = f"✅ OK ({n_unique} unique)"

        print(f"{exp_name:30s} QDS={result['qds']:.4f}  MCE={result['mce']:.4f}  {status}")

    if not results:
        print("\n⚠️  No completed experiments found!")
        return pd.DataFrame(), {}

    df = pd.DataFrame(results)

    # Analyze interaction pattern
    summary = analyze_interaction_effect(df)

    return df, summary


def analyze_interaction_effect(df: pd.DataFrame) -> Dict:
    """
    Analyze the conditioning × loss interaction effect.

    Four possible patterns:
    1. Concat universally better
    2. FiLM universally better
    3. Interaction (optimal conditioning depends on loss)
    4. No difference
    """
    if df.empty:
        return {}

    summary = {
        'total_experiments': len(df),
        'collapsed_count': df['collapsed'].sum(),
        'collapsed_rate': df['collapsed'].mean() * 100,
    }

    # Analyze each loss function
    for loss in ['tb', 'subtb', 'subtb_entropy']:
        loss_df = df[df['loss'] == loss]
        if loss_df.empty:
            continue

        # Get concat and FiLM results
        concat_df = loss_df[loss_df['conditioning'] == 'concat']
        film_df = loss_df[loss_df['conditioning'] == 'film']

        if concat_df.empty or film_df.empty:
            continue

        # Calculate mean QDS for each
        qds_concat = concat_df['qds'].mean()
        qds_film = film_df['qds'].mean()
        qds_diff = qds_film - qds_concat  # Positive means FiLM better

        # Calculate mean MCE for each
        mce_concat = concat_df['mce'].mean()
        mce_film = film_df['mce'].mean()
        mce_diff = mce_film - mce_concat

        # Calculate collapse rates
        collapse_concat = concat_df['collapsed'].sum()
        collapse_film = film_df['collapsed'].sum()

        summary[f'{loss}_qds_concat'] = qds_concat
        summary[f'{loss}_qds_film'] = qds_film
        summary[f'{loss}_qds_diff'] = qds_diff
        summary[f'{loss}_mce_concat'] = mce_concat
        summary[f'{loss}_mce_film'] = mce_film
        summary[f'{loss}_mce_diff'] = mce_diff
        summary[f'{loss}_collapsed_concat'] = collapse_concat
        summary[f'{loss}_collapsed_film'] = collapse_film

    return summary


def print_interaction_summary(df: pd.DataFrame, summary: Dict) -> None:
    """Print interaction pattern analysis."""
    print(f"\n{'='*80}")
    print("INTERACTION PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    if df.empty:
        print("No data to analyze.")
        return

    print("Possible Patterns:")
    print("  1. Concat universally better: All concat QDS > FiLM QDS")
    print("  2. FiLM universally better: All FiLM QDS > concat QDS")
    print("  3. Interaction: Optimal conditioning depends on loss function")
    print("  4. No difference: QDS differences < 0.05 for all losses")
    print()

    # Analyze each loss function
    concat_wins = 0
    film_wins = 0
    ties = 0

    for loss in ['tb', 'subtb', 'subtb_entropy']:
        loss_label = {'tb': 'TB', 'subtb': 'SubTB(λ=0.9)', 'subtb_entropy': 'SubTB+Entropy'}[loss]

        qds_concat = summary.get(f'{loss}_qds_concat', None)
        qds_film = summary.get(f'{loss}_qds_film', None)
        qds_diff = summary.get(f'{loss}_qds_diff', None)
        collapse_concat = summary.get(f'{loss}_collapsed_concat', 0)
        collapse_film = summary.get(f'{loss}_collapsed_film', 0)

        if qds_concat is None or qds_film is None:
            print(f"{loss_label:20s}: ⚠️  INCOMPLETE DATA")
            continue

        # Determine winner (threshold of 0.05 for significance)
        if abs(qds_diff) < 0.05:
            winner = "tie"
            status = "⚔️  TIE"
            ties += 1
        elif qds_diff > 0:
            winner = "film"
            status = "🎬 FiLM better"
            film_wins += 1
        else:
            winner = "concat"
            status = "📎 Concat better"
            concat_wins += 1

        # Check for collapses
        if collapse_concat > 0 or collapse_film > 0:
            status += " (COLLAPSE!)"

        print(f"{loss_label:20s}: {status}")
        print(f"  Concat: QDS={qds_concat:.4f}  collapsed={collapse_concat}")
        print(f"  FiLM:   QDS={qds_film:.4f}  collapsed={collapse_film}")
        print(f"  Diff:   {qds_diff:+.4f} (FiLM - Concat)")
        print()

    # Overall pattern
    print(f"\n{'='*80}")
    print("OVERALL PATTERN")
    print(f"{'='*80}\n")

    total = concat_wins + film_wins + ties
    if total == 0:
        print("⚠️  Insufficient data")
    elif concat_wins == total:
        print("✅ Pattern 1: Concat UNIVERSALLY better")
        print("   → Keep concat in all best_config.yaml files")
    elif film_wins == total:
        print("✅ Pattern 2: FiLM UNIVERSALLY better")
        print("   → Update all best_config.yaml files to use FiLM")
    elif ties == total:
        print("✅ Pattern 4: NO DIFFERENCE")
        print("   → Keep concat (simpler implementation)")
    else:
        print("✅ Pattern 3: INTERACTION exists")
        print("   → Optimal conditioning depends on loss function")
        print(f"   Concat wins: {concat_wins}/{total}")
        print(f"   FiLM wins:   {film_wins}/{total}")
        print(f"   Ties:        {ties}/{total}")


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
    if summary.get('collapsed_rate', 0) > 50:
        print("❌ FAILED: >50% experiments collapsed")
        print("   Config structure may be broken.")
        return False

    # Check if any conditioning caused consistent collapse
    for cond in ['concat', 'film']:
        cond_df = df[df['conditioning'] == cond]
        if not cond_df.empty:
            cond_collapsed = cond_df['collapsed'].sum()
            cond_total = len(cond_df)
            if cond_collapsed / cond_total > 0.5:
                print(f"❌ FAILED: {cond} conditioning caused >50% collapse")
                print(f"   {cond} may not be working correctly.")
                return False

    print("✅ PASSED: Config structure is correct")
    print("   - No widespread mode collapse")
    print("   - Both concat and FiLM conditioning functional")
    print("   - Ready for full validation experiment")
    return True


def save_analysis(df: pd.DataFrame, summary: Dict, output_dir: Path) -> None:
    """Save analysis results to files."""
    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    # Save detailed results
    if not df.empty:
        csv_file = analysis_dir / 'conditioning_loss_analysis.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n📊 Saved detailed analysis to: {csv_file}")

    # Save summary
    summary_file = analysis_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Saved summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Test conditioning × loss interaction validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('tests/validation/test_conditioning_loss_config.yaml'),
        help='Path to test configuration'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('tests/validation/results_conditioning_loss'),
        help='Output directory for test results'
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
