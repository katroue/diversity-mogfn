#!/usr/bin/env python3
"""
Run ablation studies for diversity mechanisms in MOGFNs.

This script automates running multiple experimental configurations
for systematic ablation studies.

Usage:
    # Capacity ablation
    python scripts/ablations/run_ablation_study.py \
        --config configs/ablations/capacity_ablation.yaml \
        --ablation capacity \
        --output_dir results/ablations/capacity
    
    # Sampling ablation
    python scripts/ablations/run_ablation_study.py \
        --config configs/ablations/sampling_ablation.yaml \
        --ablation sampling \
        --output_dir results/ablations/sampling
    
    # Loss ablation with resume after updated groups based on prior results
    python scripts/ablations/run_ablation_study.py \
        --config configs/ablations/loss_ablation_final.yaml \
        --ablation loss \
        --output_dir results/ablations/loss
        --resume

"""
import sys
import argparse
import yaml
import os
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
import torch

# Add project root to path for src imports
# __file__ is in scripts/ablations/, so go up 2 levels to project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.mogfn_pc import MOGFN_PC, PreferenceSampler, MOGFNTrainer
from src.environments.hypergrid import HyperGrid
from src.metrics.traditional import compute_all_traditional_metrics
from src.metrics.trajectory import trajectory_diversity_score, multi_path_diversity
from src.metrics.spatial import mode_coverage_entropy, pairwise_minimum_distance
from src.metrics.objective import preference_aligned_spread, pareto_front_smoothness
from src.metrics.dynamics import replay_buffer_diversity
from src.metrics.flow import flow_concentration_index
from src.metrics.composite import quality_diversity_score, diversity_efficiency_ratio
from src.utils.tensor_utils import to_numpy, to_hashable


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_name(exp_config: dict, seed: int) -> str:
    """Create unique experiment name."""
    name = exp_config.get('name', 'exp')
    return f"{name}_seed{seed}"


def run_single_experiment(exp_config: dict, 
                        fixed_config: dict,
                        seed: int,
                        output_dir: Path,
                        device: str = 'cpu') -> dict:
    """
    Run a single experimental configuration.
    
    Args:
        exp_config: Experiment-specific configuration
        fixed_config: Fixed parameters across all experiments
        seed: Random seed
        output_dir: Directory to save results
        device: Device to use
    
    Returns:
        results: Dictionary of results and metrics
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Merge configs
    config = {**fixed_config, **exp_config, 'seed': seed}
    
    # Create experiment name
    exp_name = create_experiment_name(exp_config, seed)
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}")
    
    # Create environment
    env = HyperGrid(
        height=config.get('height', 8),
        num_objectives=config.get('num_objectives', 2),
        reward_config=config.get('reward_config', 'corners')
    )
    
    # Create MOGFN model
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=config.get('hidden_dim', 128),
        num_actions=env.num_actions,
        num_layers=config.get('num_layers', 4),
        preference_encoding=config.get('preference_encoding', 'vanilla'),
        conditioning_type=config.get('conditioning', 'concat'),
        temperature=config.get('temperature', 2.0),
        sampling_strategy=config.get('sampling_strategy', 'categorical'),
        top_k=config.get('top_k', None),
        top_p=config.get('top_p', None)
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in mogfn.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create preference sampler
    pref_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution=config.get('preference_distribution', 'dirichlet'),
        alpha=config.get('dirichlet_alpha', 1.5)
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        mogfn.parameters(), 
        lr=config.get('learning_rate', 1e-3)
    )
    
    # Create trainer
    trainer = MOGFNTrainer(
      mogfn=mogfn,
      env=env,
      preference_sampler=pref_sampler,
      optimizer=optimizer,
      beta=config.get('beta', 1.0),
      off_policy_ratio=config.get('off_policy_ratio', 0.0),
      loss_function=config.get('base_loss_type', 'trajectory_balance'),
      loss_params=config.get('base_loss_params', {}),
      regularization=config.get('regularization_type', 'none'),
      regularization_params=config.get('regularization_params', {}),
      modifications=config.get('modifications_type', 'none'),
      modifications_params=config.get('modifications_params', {}),
      gradient_clip=config.get('gradient_clip', 10.0)
    )
    
    # Training
    print(f"\nTraining for {config.get('num_iterations', 10000)} iterations...")
    start_time = datetime.now()
    
    training_history = trainer.train(
        num_iterations=config.get('num_iterations', 10000),
        batch_size=config.get('batch_size', 128),
        log_every=config.get('log_every', 500)
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Evaluation
    print(f"\nEvaluating with {config.get('eval_samples', 1000)} samples...")
    eval_results = trainer.evaluate(num_samples=config.get('eval_samples', 1000))
    
    objectives_tensor = eval_results['objectives']
    preferences_tensor = eval_results['preferences']

    # Convert to numpy for metrics computation
    objectives = to_numpy(objectives_tensor)
    
    # Compute all metrics
    print("Computing metrics...")
    metrics = {}
    
    # Traditional metrics
    reference_point = np.array([1.1] * env.num_objectives)
    traditional = compute_all_traditional_metrics(objectives, reference_point)
    metrics.update(traditional)
    
    # Trajectory metrics (need to sample trajectories)
    print("  Computing trajectory metrics...")
    from src.models.mogfn_pc import MOGFNSampler
    sampler = MOGFNSampler(mogfn, env, pref_sampler)
    trajectories = []
    # FIXED: Use tensor version of preferences for PyTorch model
    for i in range(min(100, len(preferences_tensor))):  # Sample subset for efficiency
        traj = sampler.sample_trajectory(preferences_tensor[i], explore=False)
        trajectories.append(traj)
    
    metrics['tds'] = trajectory_diversity_score(trajectories)
    metrics['mpd'] = multi_path_diversity(trajectories)
    
    # Spatial metrics
    print("  Computing spatial metrics...")
    metrics['mce'], metrics['num_modes'] = mode_coverage_entropy(objectives)
    metrics['pmd'] = pairwise_minimum_distance(objectives)
    metrics['pfs'] = pareto_front_smoothness(objectives)
    
    # Objective metrics
    print("  Computing objective metrics...")
    # Compute a simplified version of Preference-Aligned Spread
    # Instead of the complex sampling, we use the already-evaluated objectives
    # and check how spread out they are for different preference groups
    try:
        # Simple alternative: measure spread within preference-conditioned groups
        # Group objectives by similar preferences
        from scipy.spatial.distance import pdist
        
        if len(objectives) > 10:  # Need enough samples
            # Simple PAS approximation: average pairwise distance in objective space
            # This is a proxy for preference-aligned spread
            dists = pdist(objectives, metric='euclidean')
            pas_approx = float(np.mean(dists))
            metrics['pas'] = pas_approx
            print(f"  ✓ PAS (simplified): {pas_approx:.4f}")
        else:
            metrics['pas'] = 0.0
            print(f"  Note: Not enough samples for PAS")
    except Exception as e:
        print(f"  Warning: Could not compute PAS: {e}")
        metrics['pas'] = 0.0
    
    # Dynamics metrics
    print("  Computing dynamics metrics...")
    metrics['rbd'] = replay_buffer_diversity(trajectories, metric='trajectory_distance')
    
    # Flow metrics
    print("  Computing flow metrics...")
    # Track state visits
    state_visits = {}
    for traj in trajectories:
        for state in traj.states:
            # Convert state to hashable key
            state_key = to_hashable(state)
            state_visits[state_key] = state_visits.get(state_key, 0) + 1
    metrics['fci'] = flow_concentration_index(state_visits)
    
    # Composite metrics
    print("  Computing composite metrics...")
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
    
    # Add configuration info
    for key in ['capacity', 'hidden_dim', 'num_layers', 'conditioning', 
                'alpha', 'preference_sampling', 'loss']:
        if key in config:
            metrics[key] = config[key]
        elif key in exp_config:
            metrics[key] = exp_config[key]
    
    # Save results
    print(f"\nSaving results to {exp_dir}...")
    
    # Save model checkpoint
    torch.save({
        'model_state_dict': mogfn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics,
    }, exp_dir / 'checkpoint.pt')
    
    # Save metrics as JSON
    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Convert preferences to numpy for saving
    preferences_numpy = to_numpy(preferences_tensor)

    # Save objectives and preferences
    np.save(exp_dir / 'objectives.npy', objectives)
    np.save(exp_dir / 'preferences.npy', preferences_numpy)
    
    # Save training history
    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"✓ Experiment completed: {exp_name}")
    print(f"  Hypervolume: {metrics['hypervolume']:.4f}")
    print(f"  TDS: {metrics['tds']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    print(f"  DER: {metrics['der']:.6f}")
    
    return metrics


def run_ablation_study(config_path: str,
                    ablation_type: str,
                    output_dir: str,
                    device: str = 'cpu',
                    resume: bool = False) -> pd.DataFrame:
    """
    Run complete ablation study.
    
    Args:
        config_path: Path to configuration YAML
        ablation_type: Type of ablation ('capacity', 'sampling', 'loss')
        output_dir: Output directory for results
        device: Device to use
        resume: Whether to resume from existing results
    
    Returns:
        results_df: DataFrame with all results
    """
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get experiment configurations
    experiments = config['experiments']
    fixed_config = config.get('fixed', {})
    seeds = fixed_config.get('seeds', [42, 123, 456, 789, 1011])
    
    print(f"\n{'='*70}")
    print(f"Ablation Study: {ablation_type}")
    print(f"{'='*70}")
    print(f"Number of configurations: {len(experiments)}")
    print(f"Number of seeds: {len(seeds)}")
    print(f"Total experiments: {len(experiments) * len(seeds)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load existing results if resuming
    results_file = output_dir / 'all_results.csv'
    if resume and results_file.exists():
        print("Resuming from existing results...")
        existing_df = pd.read_csv(results_file)
        completed_experiments = set(existing_df['exp_name'].values)
        print(f"Found {len(completed_experiments)} completed experiments")
    else:
        completed_experiments = set()
    
    # Run all experiments
    all_results = []
    total_experiments = len(experiments) * len(seeds)
    
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for exp_config in experiments:
            for seed in seeds:
                exp_name = create_experiment_name(exp_config, seed)
                
                # Skip if already completed
                if exp_name in completed_experiments:
                    print(f"Skipping completed experiment: {exp_name}")
                    pbar.update(1)
                    continue
                
                try:
                    # Run experiment
                    metrics = run_single_experiment(
                        exp_config=exp_config,
                        fixed_config=fixed_config,
                        seed=seed,
                        output_dir=output_dir,
                        device=device
                    )
                    
                    all_results.append(metrics)
                    
                    # Save intermediate results
                    if len(all_results) % 5 == 0:
                        temp_df = pd.DataFrame(all_results)
                        temp_df.to_csv(output_dir / 'results_temp.csv', index=False)
                    
                except Exception as e:
                    print(f"\n✗ Error in experiment {exp_name}: {str(e)}")
                    print("Continuing with next experiment...")
                    import traceback
                    traceback.print_exc()
                
                pbar.update(1)
    
    # Create final results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # If resuming, merge with existing results
        if resume and results_file.exists():
            existing_df = pd.read_csv(results_file)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)
        
        # Save final results
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ All results saved to {results_file}")
        
        # Print summary statistics
        print("\n" + "="*70)
        print("Summary Statistics")
        print("="*70)
        print(results_df.describe())
        
        return results_df
    else:
        print("\n✗ No new results to save")
        if resume and results_file.exists():
            return pd.read_csv(results_file)
        return pd.DataFrame()


def create_summary_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Génère des rapports sommaires et CSV groupés.
    Écrit :
      - all_results.csv (raw)
      - summary_by_capacity.csv
      - summary_by_conditioning.csv
      - summary_by_capacity_and_conditioning.csv
      - overall_summary.csv (describe)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde brute
    results_df.to_csv(output_dir / 'all_results.csv', index=False)

    # S'assurer que les colonnes de groupement existent
    for col in ('capacity', 'conditioning'):
        if col not in results_df.columns:
            results_df[col] = np.nan

    # Sélection des colonnes numériques pour l'agrégation
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        # Rien à agréger : on sauvegarde juste le dataframe complet et on sort
        results_df.to_csv(output_dir / 'all_results_no_numeric.csv', index=False)
        return

    def _agg_and_save(df, groupby_cols, out_name):
        grp = df.groupby(groupby_cols)[numeric_cols].agg(['mean', 'std', 'count'])
        # Aplatir les MultiIndex de colonnes
        grp.columns = ['_'.join(map(str, c)).strip() for c in grp.columns.values]
        grp = grp.reset_index()
        grp.to_csv(output_dir / out_name, index=False)

    # Groupements demandés
    _agg_and_save(results_df, 'capacity', 'summary_by_capacity.csv')
    _agg_and_save(results_df, 'conditioning', 'summary_by_conditioning.csv')
    _agg_and_save(results_df, ['capacity', 'conditioning'], 'summary_by_capacity_and_conditioning.csv')

    # Résumé global (describe) — convertir en DF et sauver
    overall = results_df.describe(include='all')
    try:
        overall_df = overall.transpose().reset_index().rename(columns={'index': 'metric'})
        overall_df.to_csv(output_dir / 'overall_summary.csv', index=False)
    except Exception:
        # Si describe renvoie des types non sérialisables, sauvegarder la version simple
        results_df[numeric_cols].describe().transpose().reset_index().to_csv(
            output_dir / 'overall_summary_numeric.csv', index=False
        )


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation studies for diversity mechanisms in MOGFNs'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--ablation',
        type=str,
        required=True,
        choices=['capacity', 'sampling', 'loss', 'custom'],
        help='Type of ablation study'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/ablations',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing results'
    )
    parser.add_argument(
        '--no_report',
        action='store_true',
        help='Skip creating summary report'
    )
    
    args = parser.parse_args()
    
    # Run ablation study
    results_df = run_ablation_study(
        config_path=args.config,
        ablation_type=args.ablation,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume
    )
    
    # Create summary report
    if not args.no_report and len(results_df) > 0:
        create_summary_report(
            results_df=results_df,
            ablation_type=args.ablation,
            output_dir=Path(args.output_dir)
        )
    
    print("\n" + "="*70)
    print("Ablation study completed!")
    print("="*70)


if __name__ == '__main__':
    main()