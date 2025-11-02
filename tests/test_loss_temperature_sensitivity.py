#!/usr/bin/env python3
"""
Pilot test for loss function temperature sensitivity.

Tests whether the relative ranking of loss functions changes when using
temperature=2.0 (diversity-optimized) vs temperature=5.0 (quality-optimized).

Tests 3 key loss functions:
- Trajectory Balance (TB)
- Subtrajectory Balance λ=0.9 (SubTB)
- Detailed Balance (DB)

Usage:
    python tests/test_loss_temperature_sensitivity.py
"""
import sys
import yaml
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ablations.run_ablation_study import run_single_experiment, load_config


def create_pilot_config():
    """Create a pilot test configuration with reduced iterations."""
    # Load full loss ablation config
    config_path = project_root / 'configs' / 'ablations' / 'loss_ablation_final.yaml'
    config = load_config(str(config_path))

    # Modify for quick testing
    config['fixed']['num_iterations'] = 1000  # Reduced from 4000
    config['fixed']['eval_samples'] = 500  # Reduced from 1000
    config['fixed']['eval_every'] = 250  # More frequent evaluation
    config['fixed']['batch_size'] = 64  # Smaller batch for speed
    config['fixed']['final_eval_samples'] = 2000  # Reduced from 10000
    # Don't set temperature in fixed - it will be set per experiment
    config['seeds'] = [42, 153]  # 2 seeds for statistical robustness

    # Test experiments: 3 loss functions × 2 temperatures = 6 experiments
    test_experiments = []

    loss_functions = [
        {
            'name': 'trajectory_balance',
            'label': 'TB',
            'type': 'trajectory_balance',
            'params': {'log_reward_clip': 10.0}
        },
        {
            'name': 'subtrajectory_balance_09',
            'label': 'SubTB(λ=0.9)',
            'type': 'subtrajectory_balance',
            'params': {'lambda_': 0.9, 'log_reward_clip': 10.0}
        },
        {
            'name': 'detailed_balance',
            'label': 'DB',
            'type': 'detailed_balance',
            'params': {'log_reward_clip': 10.0}
        }
    ]

    temperatures = [
        {'value': 2.0, 'label': 't2.0'},
        {'value': 5.0, 'label': 't5.0'}
    ]

    for loss in loss_functions:
        for temp in temperatures:
            exp_name = f"{loss['name']}_{temp['label']}"
            test_experiments.append({
                'name': exp_name,
                'group': 'temperature_pilot',
                'description': f"{loss['label']} with temperature={temp['value']}",
                'base_loss': loss['name'],
                'base_loss_type': loss['type'],
                'base_loss_params': loss['params'],
                'base_loss_label': loss['label'],
                'regularization': 'none',
                'regularization_type': 'none',
                'regularization_params': {},
                'modifications': 'standard',
                'modifications_type': 'none',
                'modifications_params': {},
                'temperature': temp['value']  # Override temperature
            })

    config['experiments'] = test_experiments
    return config


def run_pilot_test():
    """Run the pilot test and analyze results."""
    print("=" * 80)
    print("LOSS FUNCTION TEMPERATURE SENSITIVITY PILOT TEST")
    print("=" * 80)
    print("\nTesting 3 loss functions with temperature 2.0 vs 5.0")
    print("Loss functions: TB, SubTB(λ=0.9), DB")
    print("Seeds: [42, 153]")
    print("Iterations: 1000 (reduced from 4000 for speed)")
    print("\nThis test will take approximately 15-20 minutes...\n")

    # Create test config
    config = create_pilot_config()

    # Create temporary output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / 'results' / 'ablations' / 'loss' / f'temperature_pilot_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Save test config
    config_path = output_dir / 'pilot_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run experiments
    results = []
    total_experiments = len(config['experiments']) * len(config['seeds'])
    current = 0

    for exp in config['experiments']:
        for seed in config['seeds']:
            current += 1
            print(f"\n[{current}/{total_experiments}] Running: {exp['name']} (seed={seed})")
            print("-" * 80)

            # Run experiment
            try:
                exp_dir = output_dir / f"{exp['name']}_seed{seed}"
                result = run_single_experiment(
                    exp_config=exp,
                    fixed_config=config['fixed'],
                    seed=seed,
                    output_dir=exp_dir,
                    device='cpu'
                )

                # Store results with additional metadata
                result['exp_name'] = exp['name']
                result['loss_function'] = exp['base_loss']
                result['temperature'] = exp.get('temperature', config['fixed'].get('temperature', 2.0))
                result['seed'] = seed
                results.append(result)

                print(f"✓ Completed: {exp['name']} (seed={seed})")

            except Exception as e:
                print(f"✗ Failed: {exp['name']} (seed={seed})")
                print(f"  Error: {str(e)}")
                import traceback
                traceback.print_exc()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    results_csv = output_dir / 'pilot_results.csv'
    df.to_csv(results_csv, index=False)
    print(f"\n✓ Raw results saved to: {results_csv}")

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS: Temperature Sensitivity")
    print("=" * 80)

    analyze_temperature_sensitivity(df, output_dir)

    return df, output_dir


def analyze_temperature_sensitivity(df, output_dir):
    """Analyze whether loss function rankings change with temperature."""

    key_metrics = [
        'hypervolume',
        'mode_coverage_entropy',
        'pairwise_minimum_distance',
        'preference_aligned_spread',
        'quality_diversity_score'
    ]

    print("\n" + "=" * 80)
    print("RANKINGS BY TEMPERATURE")
    print("=" * 80)

    # Split by temperature
    df_t2 = df[df['temperature'] == 2.0].copy()
    df_t5 = df[df['temperature'] == 5.0].copy()

    # Group by loss function and aggregate
    metrics_t2 = df_t2.groupby('loss_function')[key_metrics].mean()
    metrics_t5 = df_t5.groupby('loss_function')[key_metrics].mean()

    comparison_data = []
    ranking_changes = []

    for metric in key_metrics:
        print(f"\n{metric.upper().replace('_', ' ')}")
        print("-" * 80)

        # Get rankings for both temperatures
        rank_t2 = metrics_t2[metric].rank(ascending=False)
        rank_t5 = metrics_t5[metric].rank(ascending=False)

        # Create comparison table
        comparison = pd.DataFrame({
            'Loss': metrics_t2.index,
            'T=2.0 (Value)': metrics_t2[metric].values,
            'T=2.0 (Rank)': rank_t2.values.astype(int),
            'T=5.0 (Value)': metrics_t5[metric].values,
            'T=5.0 (Rank)': rank_t5.values.astype(int),
            'Rank Change': (rank_t5 - rank_t2).values.astype(int)
        })

        comparison = comparison.sort_values('T=2.0 (Rank)')

        # Print as formatted table without tabulate
        print("\n" + "-" * 100)
        print(f"{'Loss':<25} {'T=2.0 (Value)':>15} {'T=2.0 (Rank)':>12} {'T=5.0 (Value)':>15} {'T=5.0 (Rank)':>12} {'Rank Change':>12}")
        print("-" * 100)
        for _, row in comparison.iterrows():
            print(f"{row['Loss']:<25} {row['T=2.0 (Value)']:>15.4f} {row['T=2.0 (Rank)']:>12} {row['T=5.0 (Value)']:>15.4f} {row['T=5.0 (Rank)']:>12} {row['Rank Change']:>12}")
        print("-" * 100)

        # Check if winner changed
        winner_t2 = rank_t2.idxmin()
        winner_t5 = rank_t5.idxmin()

        if winner_t2 != winner_t5:
            ranking_changes.append({
                'metric': metric,
                'winner_t2': winner_t2,
                'winner_t5': winner_t5,
                'changed': True
            })
            print(f"\n⚠️  RANKING CHANGED: {winner_t2} (T=2.0) → {winner_t5} (T=5.0)")
        else:
            ranking_changes.append({
                'metric': metric,
                'winner_t2': winner_t2,
                'winner_t5': winner_t5,
                'changed': False
            })
            print(f"\n✓ Winner consistent: {winner_t2}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n_changed = sum(r['changed'] for r in ranking_changes)
    n_total = len(ranking_changes)

    print(f"\nMetrics where ranking changed: {n_changed}/{n_total}")

    if n_changed > 0:
        print("\n⚠️  RECOMMENDATION: Temperature affects loss function rankings!")
        print("   You should re-run the loss ablation with temperature=5.0 for")
        print("   consistency with your sampling ablation results.")
    else:
        print("\n✓ RECOMMENDATION: Rankings are consistent across temperatures.")
        print("  You can continue with temperature=2.0 if preferred.")
        print("  Consider documenting this finding in your analysis.")

    # Save ranking comparison
    ranking_df = pd.DataFrame(ranking_changes)
    ranking_csv = output_dir / 'ranking_comparison.csv'
    ranking_df.to_csv(ranking_csv, index=False)
    print(f"\n✓ Ranking comparison saved to: {ranking_csv}")

    # Save aggregated metrics
    metrics_comparison = pd.DataFrame({
        'loss_function': metrics_t2.index,
        **{f'{m}_t2.0': metrics_t2[m].values for m in key_metrics},
        **{f'{m}_t5.0': metrics_t5[m].values for m in key_metrics}
    })
    metrics_csv = output_dir / 'metrics_comparison.csv'
    metrics_comparison.to_csv(metrics_csv, index=False)
    print(f"✓ Metrics comparison saved to: {metrics_csv}")


if __name__ == '__main__':
    try:
        df, output_dir = run_pilot_test()
        print("\n" + "=" * 80)
        print("PILOT TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nResults directory: {output_dir}")
        print("\nNext steps:")
        print("1. Review the ranking comparison to see if temperature affects loss rankings")
        print("2. If rankings changed: Update loss_ablation_final.yaml with temperature=5.0")
        print("3. If rankings consistent: Continue with current configuration")

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
