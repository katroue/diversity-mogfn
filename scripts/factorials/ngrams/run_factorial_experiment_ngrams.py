#!/usr/bin/env python3
"""
Run factorial experiments for Multi-Objective GFlowNets on N-grams environment.

This script runs factorial experiments that test interactions between
multiple factors (e.g., capacity x sampling temperature, capacity x loss function, sampling x loss function).

Supports all three factorial configurations:
    - capacity_sampling_2way.yaml: Model capacity � Sampling temperature
    - capacity_loss_2way.yaml: Model capacity � Loss function
    - sampling_loss_2way.yaml: Sampling temperature � Loss function

Usage:
    # Run capacity � loss factorial
    python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
        --config configs/factorials/ngrams_capacity_loss_2way.yaml \
        --output_dir results/factorials/ngrams_capacity_loss

    # Dry run to preview
    python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
        --config configs/factorials/ngrams_sampling_loss_2way.yaml \
        --output_dir results/factorials/ngrams_sampling_loss \
        --resume --dry-run

    # Resume interrupted experiment
    python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
        --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
        --resume

    # Run specific conditions only
    python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
        --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
        --conditions small_low,medium_high
"""

import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_factorial_config(config_path: Path) -> dict:
    """Load factorial experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_condition(config: dict, condition: dict) -> dict:
    """
    Parse a condition definition to get full experiment parameters.

    Args:
        config: Full factorial configuration
        condition: Single condition specification (e.g., {'capacity': 'small', 'temperature': 'low'})

    Returns:
        Full experiment configuration for this condition
    """
    exp_config = config['fixed'].copy()

    # Add name
    exp_config['condition_name'] = condition['name']

    # Parse each factor level and merge parameters
    for factor_name, level_name in condition.items():
        if factor_name == 'name':
            continue

        # Get factor definition
        factor = config['factors'][factor_name]
        level = factor['levels'][level_name]

        # Merge level parameters into experiment config
        for param_key, param_value in level.items():
            if param_key not in ['label', 'description']:
                exp_config[param_key] = param_value

        # Store factor level for tracking
        exp_config[f'factor_{factor_name}'] = level_name
        exp_config[f'{factor_name}_level'] = level_name

    return exp_config


def generate_all_conditions(config: dict) -> List[dict]:
    """
    Generate all experimental conditions from factorial design.

    If conditions are explicitly listed in config, use those.
    Otherwise, generate all combinations from factors.
    """
    if 'conditions' in config:
        # Use explicitly defined conditions
        return config['conditions']

    # Generate all combinations from factors
    factors = config['factors']
    factor_names = list(factors.keys())

    # Get all levels for each factor
    factor_levels = {}
    for factor_name, factor_def in factors.items():
        factor_levels[factor_name] = list(factor_def['levels'].keys())

    # Generate all combinations
    conditions = []
    for combination in itertools.product(*[factor_levels[f] for f in factor_names]):
        condition = {
            'name': '_'.join(combination),
        }
        for factor_name, level_name in zip(factor_names, combination):
            condition[factor_name] = level_name
        conditions.append(condition)

    return conditions


def create_experiment_configs(config: dict,
                            conditions: List[dict],
                            seeds: List[int]) -> List[Dict]:
    """
    Create individual experiment configurations for all conditions and seeds.

    Returns:
        List of experiment configurations ready to run
    """
    exp_configs = []

    for condition in conditions:
        # Parse condition to get full parameters
        condition_config = parse_condition(config, condition)
        condition_name = condition['name']

        # Create config for each seed
        for seed in seeds:
            exp_config = condition_config.copy()
            exp_config['seed'] = seed
            exp_config['exp_name'] = f"{condition_name}_seed{seed}"
            exp_configs.append(exp_config)

    return exp_configs


def run_single_experiment(exp_config: dict,
                        output_dir: Path,
                        config: dict,
                        device: str = 'cpu') -> dict:
    """
    Run a single experiment configuration on N-grams environment.

    This function handles multiple factorial types (capacity_sampling, capacity_loss, sampling_loss)
    by using .get() with sensible defaults for parameters that may not be present in all configs.

    Default values:
        - temperature: 2.0
        - sampling_strategy: 'categorical'
        - conditioning: 'concat'
        - preference_distribution: 'dirichlet'
        - dirichlet_alpha: 1.5
        - learning_rate: 0.001
        - loss_function: 'trajectory_balance'
        - gradient_clip: 10.0
        - max_iterations: 5000
        - batch_size: 128
        - num_preferences_per_batch: 16
        - final_eval_samples: 10000
        - vocab_size: 4
        - seq_length: 8
        - ngram_length: 2

    Args:
        exp_config: Experiment configuration
        output_dir: Output directory for this experiment
        config: Full factorial configuration (unused, kept for compatibility)
        device: Device to use for training

    Returns:
        Dictionary of results/metrics
    """
    import torch
    from datetime import datetime

    # Import training modules
    try:
        from src.models.mogfn_pc import MOGFN_PC, PreferenceSampler, MOGFNTrainer, MOGFNSampler
        from src.environments.ngrams import NGrams
        from src.metrics.traditional import compute_all_traditional_metrics
        from src.metrics.trajectory import trajectory_diversity_score, multi_path_diversity
        from src.metrics.spatial import mode_coverage_entropy, pairwise_minimum_distance
        from src.metrics.objective import preference_aligned_spread, pareto_front_smoothness
        from src.metrics.dynamics import replay_buffer_diversity
        from src.metrics.flow import flow_concentration_index
        from src.metrics.composite import quality_diversity_score, diversity_efficiency_ratio
        from src.utils.tensor_utils import to_numpy, to_hashable
    except ImportError as e:
        print(f"Warning: Could not import training modules: {e}")
        print("Running in placeholder mode (for testing script logic)")

        # Placeholder results for testing
        return {
            'exp_name': exp_config['exp_name'],
            'condition_name': exp_config['condition_name'],
            'seed': exp_config['seed'],
            'hypervolume': np.random.rand(),
            'tds': np.random.rand(),
            'mce': np.random.rand(),
            'pas': np.random.rand(),
            'qds': np.random.rand(),
            'status': 'placeholder',
        }

    # Set random seeds
    seed = exp_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    exp_name = exp_config['exp_name']

    # Create N-grams environment
    env = NGrams(
        vocab_size=exp_config.get('vocab_size', 4),
        seq_length=exp_config.get('seq_length', 8),
        ngram_length=exp_config.get('ngram_length', 2),
        objective_patterns=exp_config.get('objective_patterns', None),
        normalize_rewards=exp_config.get('normalize_rewards', True)
    )

    print(f"\n  Environment: N-grams")
    print(f"    Vocabulary size: {env.vocab_size}")
    print(f"    Sequence length: {env.seq_length}")
    print(f"    N-gram length: {env.ngram_length}")
    print(f"    Num objectives: {env.num_objectives}")
    print(f"    Patterns: {env.objective_patterns}")

    # Create MOGFN model
    # Use .get() with defaults for parameters that may not be in all factorial configs
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=exp_config['hidden_dim'],
        num_actions=env.num_actions,
        num_layers=exp_config['num_layers'],
        preference_encoding='vanilla',
        conditioning_type=exp_config.get('conditioning', 'concat'),
        temperature=exp_config.get('temperature', 2.0),  # Default to 1.0 if not specified
        sampling_strategy=exp_config.get('sampling_strategy', 'categorical')  # Default to categorical
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in mogfn.parameters())

    # Create preference sampler
    # Use .get() with defaults for parameters that may not be in all factorial configs
    pref_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution=exp_config.get('preference_distribution', 'dirichlet'),
        alpha=exp_config.get('dirichlet_alpha', 1.5)
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        mogfn.parameters(),
        lr=exp_config.get('learning_rate', 0.001)
    )

    # Create trainer
    trainer = MOGFNTrainer(
        mogfn=mogfn,
        env=env,
        preference_sampler=pref_sampler,
        optimizer=optimizer,
        loss_function=exp_config.get('loss_function', 'trajectory_balance'),
        loss_params=exp_config.get('loss_params', {}),
        regularization=exp_config.get('regularization', 'none'),
        regularization_params=exp_config.get('regularization_params', {}),
        modifications=exp_config.get('modifications', 'none'),
        modifications_params=exp_config.get('modifications_params', {}),
        gradient_clip=exp_config.get('gradient_clip', 10.0)
    )

    # Training
    start_time = datetime.now()

    training_history = trainer.train(
        num_iterations=exp_config.get('max_iterations', 5000),
        batch_size=exp_config.get('batch_size', 128),
        num_preferences_per_batch=exp_config.get('num_preferences_per_batch', 16),
        log_every=exp_config.get('eval_every', 500)
    )

    training_time = (datetime.now() - start_time).total_seconds()

    # Evaluation
    eval_results = trainer.evaluate(num_samples=exp_config.get('final_eval_samples', 10000))

    objectives_tensor = eval_results['objectives']
    preferences_tensor = eval_results['preferences']
    objectives = to_numpy(objectives_tensor)

    # Compute metrics
    metrics = {}

    # Traditional metrics
    # For N-grams, reference point depends on normalization
    if env.normalize_rewards:
        reference_point = np.array([1.1] * env.num_objectives)
    else:
        reference_point = np.array([env.max_count + 1.0] * env.num_objectives)

    traditional = compute_all_traditional_metrics(objectives, reference_point)
    metrics.update(traditional)

    # Trajectory metrics
    sampler = MOGFNSampler(mogfn, env, pref_sampler)
    trajectories = []
    for i in range(min(100, len(preferences_tensor))):
        traj = sampler.sample_trajectory(preferences_tensor[i], explore=False)
        trajectories.append(traj)

    metrics['tds'] = trajectory_diversity_score(trajectories)
    metrics['mpd'] = multi_path_diversity(trajectories)

    # Spatial metrics
    metrics['mce'], metrics['num_modes'] = mode_coverage_entropy(objectives)
    metrics['pmd'] = pairwise_minimum_distance(objectives)
    metrics['pfs'] = pareto_front_smoothness(objectives)

    # Objective metrics (simplified PAS)
    try:
        from scipy.spatial.distance import pdist
        if len(objectives) > 10:
            dists = pdist(objectives, metric='euclidean')
            metrics['pas'] = float(np.mean(dists))
        else:
            metrics['pas'] = 0.0
    except Exception:
        metrics['pas'] = 0.0

    # Dynamics metrics
    metrics['rbd'] = replay_buffer_diversity(trajectories, metric='trajectory_distance')

    # Flow metrics
    state_visits = {}
    for traj in trajectories:
        for state in traj.states:
            state_key = to_hashable(state)
            state_visits[state_key] = state_visits.get(state_key, 0) + 1
    metrics['fci'] = flow_concentration_index(state_visits)

    # Composite metrics
    qds_results = quality_diversity_score(
        objectives,
        reference_point,
        alpha=0.5
    )
    metrics['qds'] = qds_results['qds']

    der_results = diversity_efficiency_ratio(
        objectives,
        training_time=training_time,
        num_parameters=num_params
    )
    metrics['der'] = der_results['der']

    # Add metadata
    metrics['num_parameters'] = num_params
    metrics['training_time'] = training_time
    metrics['final_loss'] = training_history['loss'][-1] if training_history['loss'] else 0.0
    metrics['seed'] = seed
    metrics['exp_name'] = exp_name
    metrics['condition_name'] = exp_config['condition_name']

    # Save results
    # Save model checkpoint
    torch.save({
        'model_state_dict': mogfn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': exp_config,
        'metrics': metrics,
    }, output_dir / 'checkpoint.pt')

    # Save metrics as JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save objectives and preferences
    preferences_numpy = to_numpy(preferences_tensor)
    np.save(output_dir / 'objectives.npy', objectives)
    np.save(output_dir / 'preferences.npy', preferences_numpy)

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    return metrics


def run_factorial_experiment(config_path: Path,
                            output_dir: Path,
                            dry_run: bool = False,
                            resume: bool = False,
                            conditions_filter: Optional[List[str]] = None,
                            device: str = 'cpu') -> None:
    """
    Run complete factorial experiment for N-grams environment.

    Args:
        config_path: Path to factorial configuration YAML
        output_dir: Output directory for results
        dry_run: If True, only print what would be run
        resume: If True, skip already completed experiments
        conditions_filter: If provided, only run these conditions
        device: Device to use for training ('cpu' or 'cuda')
    """
    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = load_factorial_config(config_path)

    experiment_name = config['experiment_name']
    study_type = config.get('study_type', 'factorial')

    print(f"\n{'='*80}")
    print(f"FACTORIAL EXPERIMENT (N-GRAMS): {experiment_name}")
    print(f"Study Type: {study_type}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all conditions
    all_conditions = generate_all_conditions(config)
    print(f"Total conditions in design: {len(all_conditions)}")

    # Filter conditions if specified
    if conditions_filter:
        all_conditions = [c for c in all_conditions if c['name'] in conditions_filter]
        print(f"Filtered to {len(all_conditions)} conditions: {conditions_filter}")

    # Get seeds
    if 'seeds' in config:
        seeds = config['seeds']
    else:
        num_seeds = config['fixed'].get('num_seeds', 5)
        base_seed = config['fixed'].get('base_seed', 42)
        seeds = [base_seed + i * 111 for i in range(num_seeds)]

    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(all_conditions)} conditions � {len(seeds)} seeds = {len(all_conditions) * len(seeds)} runs")

    # Generate experiment configurations
    exp_configs = create_experiment_configs(config, all_conditions, seeds)
    print(f"\nGenerated {len(exp_configs)} experiment configurations")

    # Check for existing results if resuming
    completed = set()
    results_file = output_dir / 'results.csv'
    results_temp_file = output_dir / 'results_temp.csv'

    if resume:
        if results_file.exists():
            print(f"\nResuming from existing results: {results_file}")
            df = pd.read_csv(results_file)
            completed = set(df['exp_name'].values)
            print(f"Found {len(completed)} completed experiments")
        elif results_temp_file.exists():
            print(f"\nResuming from temporary results: {results_temp_file}")
            df = pd.read_csv(results_temp_file)
            completed = set(df['exp_name'].values)
            print(f"Found {len(completed)} completed experiments")

    # Save configuration
    config_save_path = output_dir / 'experiment_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'conditions': all_conditions,
            'seeds': seeds,
        }, f)
    print(f"Saved configuration to: {config_save_path}")

    # Dry run: just print what would be executed
    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN - Would execute the following experiments:")
        print(f"{'='*80}\n")

        for i, exp_config in enumerate(exp_configs, 1):
            exp_name = exp_config['exp_name']
            status = "[SKIP - COMPLETED]" if exp_name in completed else "[RUN]"
            condition = exp_config['condition_name']
            seed = exp_config['seed']

            # Show factor levels
            factors_str = []
            for key, value in exp_config.items():
                if key.endswith('_level'):
                    factor = key.replace('_level', '')
                    factors_str.append(f"{factor}={value}")

            print(f"  {i:3d}. {status} {exp_name}")
            print(f"       Condition: {condition}")
            print(f"       Factors: {', '.join(factors_str)}")
            print(f"       Seed: {seed}")
            print()

        to_run = len([e for e in exp_configs if e['exp_name'] not in completed])
        print(f"\nTotal to run: {to_run}")
        print(f"Total to skip: {len(completed)}")
        return

    # Run experiments
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*80}\n")

    results = []
    failed = []

    for exp_config in tqdm(exp_configs, desc="Running experiments"):
        exp_name = exp_config['exp_name']

        # Skip if already completed
        if exp_name in completed:
            tqdm.write(f"   Skipping {exp_name} (already completed)")
            continue

        # Create experiment directory
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment config
        exp_config_file = exp_dir / 'config.json'
        with open(exp_config_file, 'w') as f:
            json.dump(exp_config, f, indent=2)

        tqdm.write(f"  Running {exp_name}...")

        try:
            # Run experiment
            result = run_single_experiment(exp_config, exp_dir, config, device)

            # Add factor levels to results
            for key, value in exp_config.items():
                if key.endswith('_level') or key in ['condition_name', 'seed']:
                    result[key] = value

            results.append(result)

            # Save incremental results
            df_temp = pd.DataFrame(results)

            # Merge with existing temp results if they exist
            if results_temp_file.exists():
                try:
                    df_existing = pd.read_csv(results_temp_file)
                    df_temp = pd.concat([df_existing, df_temp], ignore_index=True)
                    # Remove duplicates (keep last occurrence in case of re-runs)
                    df_temp = df_temp.drop_duplicates(subset=['exp_name'], keep='last')
                except Exception as e:
                    # If reading fails, just save the new results
                    tqdm.write(f"   ⚠ Warning: Could not merge with existing temp results: {e}")

            df_temp.to_csv(results_temp_file, index=False)

            tqdm.write(f"   Completed {exp_name}")

        except Exception as e:
            tqdm.write(f"   Failed {exp_name}: {str(e)}")
            failed.append({
                'exp_name': exp_name,
                'condition': exp_config['condition_name'],
                'seed': exp_config['seed'],
                'error': str(e),
            })

    # Save final results
    if results:
        df_results = pd.DataFrame(results)

        # Merge with existing results if resuming
        if resume and results_file.exists():
            df_existing = pd.read_csv(results_file)
            df_results = pd.concat([df_existing, df_results], ignore_index=True)

        df_results.to_csv(results_file, index=False)
        print(f"\n Saved results to: {results_file}")

        # Remove temp file
        if results_temp_file.exists():
            results_temp_file.unlink()

    # Save failed experiments
    if failed:
        failed_file = output_dir / 'failed.json'
        with open(failed_file, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"� {len(failed)} experiments failed, see {failed_file}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"  Total experiments: {len(exp_configs)}")
    print(f"  Completed: {len(results)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped (already done): {len(completed)}")
    print(f"{'='*80}\n")

    # Show summary by condition if results available
    if results and len(results) > 0:
        print("\n=� Results by Condition (mean � std):\n")
        df_results = pd.DataFrame(results)

        # Group by condition
        if 'condition_name' in df_results.columns:
            summary = df_results.groupby('condition_name').agg({
                'mce': ['mean', 'std'],
                'tds': ['mean', 'std'],
                'hypervolume': ['mean', 'std'],
                'qds': ['mean', 'std'],
            }).round(4)
            print(summary)

        print(f"\nFull results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run factorial experiments for Multi-Objective GFlowNets on N-grams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to factorial configuration YAML file'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Output directory for results (default: inferred from config)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be run without executing'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing results (skip completed experiments)'
    )

    parser.add_argument(
        '--conditions',
        type=str,
        default=None,
        help='Comma-separated list of conditions to run (e.g., "small_low,medium_high")'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for training (cpu or cuda)'
    )

    args = parser.parse_args()

    # Parse conditions filter
    conditions_filter = None
    if args.conditions:
        conditions_filter = [c.strip() for c in args.conditions.split(',')]

    # Determine output directory
    if args.output_dir is None:
        # Infer from config path
        config_name = args.config.stem
        args.output_dir = Path('results') / 'factorials' / config_name

    # Run factorial experiment
    run_factorial_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        conditions_filter=conditions_filter,
        device=args.device,
    )


if __name__ == '__main__':
    main()