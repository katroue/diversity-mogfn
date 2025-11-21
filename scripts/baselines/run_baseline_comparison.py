#!/usr/bin/env python3
"""
Run baseline comparison experiments.

Compares MOGFN-PC against baseline algorithms:
- Random Sampling
- NSGA-II
- HN-GFN (Hypernetwork-GFlowNet)
- (Future) Single-Objective GFlowNet

Usage:
    # Run all baselines on HyperGrid
    python scripts/baselines/run_baseline_comparison.py \
        --task sequences \
        --algorithms random,mogfn_pc,nsga2,hngfn \
        --seeds 42,153,264,375,486 \
        --output_dir results/baselines/sequences \
        --eval_samples 1000

    # Run only Random and NSGA-II (quick test)
    python scripts/baselines/run_baseline_comparison.py \
        --task hypergrid \
        --algorithms random,nsga2 \
        --seeds 42 \
        --num_iterations 1000 \
        --output_dir results/baselines/test

    # Compare MOGFN-PC and HN-GFN
    python scripts/baselines/run_baseline_comparison.py \
        --task hypergrid \
        --algorithms mogfn_pc,hngfn \
        --seeds 42,153,264,375,486 \
        --output_dir results/baselines/mogfn_vs_hngfn

Task-specific HN-GFN examples:
    # HyperGrid (4000 iterations, default architecture)
    sudo nice -10 python scripts/baselines/run_baseline_comparison.py \
        --task sequences \
        --algorithms hngfn \
        --seeds 42,153,264,375,486 \
        --output_dir results/baselines/sequences

"""

import sys
import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import RandomSampler, NSGA2Adapter, HN_GFN
from src.models.mogfn_pc import MOGFN_PC, PreferenceSampler, MOGFNTrainer, MOGFNSampler
from src.environments.hypergrid import HyperGrid
from src.environments.ngrams import NGrams
from src.environments.molecules import MoleculeFragments
from src.environments.sequences import DNASequence
from src.metrics.traditional import compute_all_traditional_metrics
from src.metrics.trajectory import trajectory_diversity_score, multi_path_diversity
from src.metrics.spatial import mode_coverage_entropy, pairwise_minimum_distance
from src.metrics.objective import pareto_front_smoothness
from src.metrics.dynamics import replay_buffer_diversity
from src.metrics.flow import flow_concentration_index
from src.metrics.composite import quality_diversity_score, diversity_efficiency_ratio
from src.utils.tensor_utils import to_numpy, to_hashable
from scipy.spatial.distance import pdist
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_environment(task: str, **kwargs):
    """Create environment based on task name."""
    if task == 'hypergrid':
        return HyperGrid(
            height=kwargs.get('height', 8),
            num_objectives=kwargs.get('num_objectives', 2),
            reward_config=kwargs.get('reward_config', 'corners')
        )
    elif task == 'ngrams':
        return NGrams(
            vocab_size=kwargs.get('vocab_size', 4),
            seq_length=kwargs.get('seq_length', 8),
        )
    elif task == 'molecules':
        return MoleculeFragments(
        )
    elif task == 'sequences':
        return DNASequence(
            
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def compute_baseline_metrics(
    objectives: np.ndarray,
    env,
    training_time: float,
    num_params: int = 0,
    algorithm: str = 'baseline'
) -> Dict[str, Any]:
    """
    Compute metrics for baseline algorithms.

    Note: Only computes metrics applicable to non-GFlowNet baselines.
    Trajectory and flow metrics require GFlowNet-specific data.

    Args:
        objectives: Objective values, shape (N, num_objectives)
        env: Environment instance
        training_time: Training time in seconds
        num_params: Number of model parameters (0 for non-parametric)
        algorithm: Algorithm name

    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}

    # Subsample if too many objectives (for computational efficiency)
    original_size = len(objectives)
    MAX_SAMPLES_FOR_METRICS = 5000  # Limit for expensive O(n^2) operations

    if original_size > MAX_SAMPLES_FOR_METRICS:
        logger.warning(
            f"  Too many samples ({original_size}) for efficient metric computation. "
            f"Subsampling to {MAX_SAMPLES_FOR_METRICS} for metrics..."
        )
        # Random subsample
        indices = np.random.choice(original_size, size=MAX_SAMPLES_FOR_METRICS, replace=False)
        objectives_for_metrics = objectives[indices]
    else:
        objectives_for_metrics = objectives

    # Traditional metrics
    logger.info(f"  Computing traditional metrics for {len(objectives_for_metrics)} samples...")
    reference_point = np.array([1.1] * env.num_objectives)
    traditional = compute_all_traditional_metrics(objectives_for_metrics, reference_point)
    metrics.update(traditional)

    # Spatial metrics
    if len(objectives_for_metrics) >= 10:
        logger.info("  Computing spatial metrics (MCE, PMD, PFS)...")
        metrics['mce'], metrics['num_modes'] = mode_coverage_entropy(objectives_for_metrics)
        metrics['pmd'] = pairwise_minimum_distance(objectives_for_metrics)
        metrics['pfs'] = pareto_front_smoothness(objectives_for_metrics)
        metrics['num_unique_solutions'] = len(np.unique(objectives, axis=0))  # Use full set for uniqueness
    else:
        logger.warning(f"Not enough samples ({len(objectives_for_metrics)}) for spatial metrics")
        metrics['mce'] = 0.0
        metrics['num_modes'] = 0
        metrics['pmd'] = 0.0
        metrics['pfs'] = 0.0
        metrics['num_unique_solutions'] = len(objectives)

    # Simplified PAS (preference-aligned spread approximation)
    if len(objectives_for_metrics) > 10:
        logger.info(f"  Computing PAS (pairwise distances)...")
        dists = pdist(objectives_for_metrics, metric='euclidean')
        metrics['pas'] = float(np.mean(dists))
        logger.info("  PAS computed")
    else:
        metrics['pas'] = 0.0

    # Composite metrics
    logger.info("  Computing composite metrics (QDS, DER)...")
    qds_results = quality_diversity_score(
        objectives_for_metrics,
        reference_point,
        alpha=0.5
    )
    metrics['qds'] = qds_results['qds']

    der_results = diversity_efficiency_ratio(
        objectives_for_metrics,
        training_time=training_time,
        num_parameters=num_params
    )
    metrics['der'] = der_results['der']

    # Metadata
    metrics['num_parameters'] = num_params
    metrics['training_time'] = training_time
    metrics['algorithm'] = algorithm
    metrics['num_samples'] = original_size  # Report original sample count
    metrics['num_samples_for_metrics'] = len(objectives_for_metrics)  # Also report subsampled count

    return metrics


def run_random_baseline(
    env,
    seed: int,
    num_iterations: int,
    batch_size: int,
    output_dir: Path
) -> Dict[str, Any]:
    """Run Random Sampling baseline."""
    logger.info(f"Running Random Sampling (seed={seed})")

    exp_dir = output_dir / f"random_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize sampler
    sampler = RandomSampler(env, max_steps=100, seed=seed)

    # "Train" (sample trajectories)
    start_time = datetime.now()
    history = sampler.train(
        num_iterations=num_iterations,
        batch_size=batch_size,
        log_interval=max(1, num_iterations // 10)
    )
    training_time = (datetime.now() - start_time).total_seconds()

    # Get objectives
    objectives = sampler.get_all_objectives()
    logger.info(f"Computing metrics for {len(objectives)} samples...")

    # Compute metrics
    metrics = compute_baseline_metrics(
        objectives,
        env,
        training_time,
        num_params=0,
        algorithm='random'
    )
    metrics['seed'] = seed
    logger.info("Metrics computed successfully")

    # Save results
    logger.info(f"Saving results to {exp_dir}...")
    np.save(exp_dir / 'objectives.npy', objectives)

    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Random baseline complete: {len(objectives)} solutions, HV={metrics.get('hypervolume', 0):.4f}")

    return metrics


def run_nsga2_baseline(
    env,
    seed: int,
    num_iterations: int,
    pop_size: int,
    output_dir: Path,
    log_interval: int = None
) -> Dict[str, Any]:
    """Run NSGA-II baseline."""
    logger.info(f"Running NSGA-II (seed={seed})")

    exp_dir = output_dir / f"nsga2_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize NSGA-II
    try:
        nsga2 = NSGA2Adapter(
            env,
            pop_size=pop_size,
            max_steps=100,
            seed=seed
        )
    except ImportError as e:
        logger.error(f"NSGA-II requires pymoo: {e}")
        logger.error("Install with: pip install pymoo")
        return {}

    # Train
    start_time = datetime.now()
    if log_interval is None:
        log_interval = max(1, num_iterations // 10)

    logger.info(f"Starting NSGA-II training ({num_iterations} generations, logging every {log_interval})...")
    logger.info("This may take several minutes for large num_iterations...")

    history = nsga2.train(
        num_iterations=num_iterations,
        log_interval=log_interval
    )
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"NSGA-II training completed in {training_time:.1f} seconds")

    # Get objectives
    objectives = nsga2.get_all_objectives()
    pareto_front = nsga2.get_pareto_front()

    # Compute metrics
    metrics = compute_baseline_metrics(
        objectives,
        env,
        training_time,
        num_params=0,
        algorithm='nsga2'
    )
    metrics['seed'] = seed
    metrics['pop_size'] = pop_size
    metrics['pareto_size'] = len(pareto_front)

    # Save results
    np.save(exp_dir / 'objectives.npy', objectives)
    np.save(exp_dir / 'pareto_front.npy', pareto_front)

    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(
        f"NSGA-II complete: {len(objectives)} solutions, "
        f"{len(pareto_front)} Pareto optimal, HV={metrics.get('hypervolume', 0):.4f}"
    )

    return metrics


def run_mogfn_pc_baseline(
    env,
    seed: int,
    num_iterations: int,
    batch_size: int,
    output_dir: Path,
    hidden_dim: int = 128,
    num_layers: int = 4,
    eval_samples: int = 1000,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run MOGFN-PC baseline."""
    logger.info(f"Running MOGFN-PC (seed={seed})")

    exp_dir = output_dir / f"mogfn_pc_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create MOGFN model
    logger.info(f"Creating MOGFN-PC model (hidden_dim={hidden_dim}, num_layers={num_layers})...")
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=hidden_dim,
        num_actions=env.num_actions,
        num_layers=num_layers,
        preference_encoding='vanilla',
        conditioning_type='film',
        temperature=2.0,
        sampling_strategy='categorical'
    ).to(device)

    num_params = sum(p.numel() for p in mogfn.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create preference sampler
    pref_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution='dirichlet',
        alpha=1.5
    )

    # Create optimizer
    optimizer = torch.optim.Adam(mogfn.parameters(), lr=1e-3)

    # Create trainer
    trainer = MOGFNTrainer(
        mogfn=mogfn,
        env=env,
        preference_sampler=pref_sampler,
        optimizer=optimizer,
        beta=1.0,
        off_policy_ratio=0.0,
        loss_function='trajectory_balance',
        loss_params={},
        regularization='none',
        regularization_params={},
        modifications='none',
        modifications_params={},
        gradient_clip=10.0
    )

    # Train
    logger.info(f"Starting MOGFN-PC training ({num_iterations} iterations)...")
    start_time = datetime.now()

    training_history = trainer.train(
        num_iterations=num_iterations,
        batch_size=batch_size,
        log_every=max(100, num_iterations // 10)
    )

    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"MOGFN-PC training completed in {training_time:.1f} seconds")

    # Evaluate
    logger.info(f"Evaluating with {eval_samples} samples...")
    eval_results = trainer.evaluate(num_samples=eval_samples)

    objectives_tensor = eval_results['objectives']
    preferences_tensor = eval_results['preferences']
    objectives = to_numpy(objectives_tensor)

    # Compute all metrics (including GFlowNet-specific ones)
    logger.info("Computing all metrics (including trajectory/flow metrics)...")
    metrics = {}

    # Subsample if needed
    original_size = len(objectives)
    MAX_SAMPLES_FOR_METRICS = 5000

    if original_size > MAX_SAMPLES_FOR_METRICS:
        logger.warning(f"  Subsampling {original_size} to {MAX_SAMPLES_FOR_METRICS} for metrics...")
        indices = np.random.choice(original_size, size=MAX_SAMPLES_FOR_METRICS, replace=False)
        objectives_for_metrics = objectives[indices]
        preferences_for_sampling = preferences_tensor[indices]
    else:
        objectives_for_metrics = objectives
        preferences_for_sampling = preferences_tensor

    # Traditional metrics
    logger.info("  Computing traditional metrics...")
    reference_point = np.array([1.1] * env.num_objectives)
    traditional = compute_all_traditional_metrics(objectives_for_metrics, reference_point)
    metrics.update(traditional)

    # Trajectory metrics
    logger.info("  Computing trajectory metrics (sampling trajectories)...")
    sampler = MOGFNSampler(mogfn, env, pref_sampler)
    trajectories = []
    num_traj_samples = min(100, len(preferences_for_sampling))
    for i in range(num_traj_samples):
        traj = sampler.sample_trajectory(preferences_for_sampling[i], explore=False)
        trajectories.append(traj)

    metrics['tds'] = trajectory_diversity_score(trajectories)
    metrics['mpd'] = multi_path_diversity(trajectories)

    # Spatial metrics
    logger.info("  Computing spatial metrics...")
    metrics['mce'], metrics['num_modes'] = mode_coverage_entropy(objectives_for_metrics)
    metrics['pmd'] = pairwise_minimum_distance(objectives_for_metrics)
    metrics['pfs'] = pareto_front_smoothness(objectives_for_metrics)
    metrics['num_unique_solutions'] = len(np.unique(objectives, axis=0))

    # Objective metrics (simplified PAS)
    logger.info("  Computing PAS...")
    if len(objectives_for_metrics) > 10:
        dists = pdist(objectives_for_metrics, metric='euclidean')
        metrics['pas'] = float(np.mean(dists))
    else:
        metrics['pas'] = 0.0

    # Dynamics metrics
    logger.info("  Computing dynamics metrics...")
    metrics['rbd'] = replay_buffer_diversity(trajectories, metric='trajectory_distance')

    # Flow metrics
    logger.info("  Computing flow metrics...")
    state_visits = {}
    for traj in trajectories:
        for state in traj.states:
            state_key = to_hashable(state)
            state_visits[state_key] = state_visits.get(state_key, 0) + 1
    metrics['fci'] = flow_concentration_index(state_visits)

    # Composite metrics
    logger.info("  Computing composite metrics...")
    qds_results = quality_diversity_score(objectives_for_metrics, reference_point, alpha=0.5)
    metrics['qds'] = qds_results['qds']

    der_results = diversity_efficiency_ratio(
        objectives_for_metrics,
        training_time=training_time,
        num_parameters=num_params
    )
    metrics['der'] = der_results['der']

    # Metadata
    metrics['num_parameters'] = num_params
    metrics['training_time'] = training_time
    metrics['final_loss'] = training_history['loss'][-1] if training_history['loss'] else 0.0
    metrics['algorithm'] = 'mogfn_pc'
    metrics['num_samples'] = original_size
    metrics['num_samples_for_metrics'] = len(objectives_for_metrics)
    metrics['seed'] = seed
    metrics['hidden_dim'] = hidden_dim
    metrics['num_layers'] = num_layers

    # Save results
    logger.info(f"Saving results to {exp_dir}...")
    torch.save({
        'model_state_dict': mogfn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, exp_dir / 'checkpoint.pt')

    np.save(exp_dir / 'objectives.npy', objectives)
    np.save(exp_dir / 'preferences.npy', to_numpy(preferences_tensor))

    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    logger.info(f"MOGFN-PC complete: {len(objectives)} solutions, HV={metrics.get('hypervolume', 0):.4f}")

    return metrics


def run_hngfn_baseline(
    env,
    seed: int,
    num_iterations: int,
    batch_size: int,
    output_dir: Path,
    hidden_dim: int = 128,
    num_layers: int = 4,
    z_hidden_dim: int = 64,
    z_num_layers: int = 3,
    eval_samples: int = 1000,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run HN-GFN (Hypernetwork-GFlowNet) baseline."""
    logger.info(f"Running HN-GFN (seed={seed})")

    exp_dir = output_dir / f"hngfn_seed{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create HN-GFN model
    logger.info(
        f"Creating HN-GFN model (hidden_dim={hidden_dim}, num_layers={num_layers}, "
        f"z_hidden_dim={z_hidden_dim}, z_num_layers={z_num_layers})..."
    )
    hngfn = HN_GFN(
        env=env,
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=hidden_dim,
        num_actions=env.num_actions,
        num_layers=num_layers,
        z_hidden_dim=z_hidden_dim,
        z_num_layers=z_num_layers,
        preference_encoding='vanilla',
        conditioning_type='film',
        learning_rate=1e-3,
        z_learning_rate=1e-3,
        alpha=1.5,
        max_steps=100,
        temperature=2.0,
        seed=seed
    )

    # Count parameters
    num_params = (
        sum(p.numel() for p in hngfn.model.parameters()) +
        sum(p.numel() for p in hngfn.Z_network.parameters())
    )
    logger.info(f"Model parameters: {num_params:,} (policy + Z hypernetwork)")

    # Train
    logger.info(f"Starting HN-GFN training ({num_iterations} iterations)...")
    start_time = datetime.now()

    training_history = hngfn.train(
        num_iterations=num_iterations,
        batch_size=batch_size,
        log_interval=max(100, num_iterations // 10)
    )

    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"HN-GFN training completed in {training_time:.1f} seconds")

    # Evaluate (sample with diverse preferences)
    logger.info(f"Evaluating with {eval_samples} samples...")
    objectives, states = hngfn.sample(num_samples=eval_samples)

    # Compute all metrics (including GFlowNet-specific ones)
    logger.info("Computing all metrics (including trajectory/flow metrics)...")
    metrics = {}

    # Subsample if needed
    original_size = len(objectives)
    MAX_SAMPLES_FOR_METRICS = 5000

    if original_size > MAX_SAMPLES_FOR_METRICS:
        logger.warning(f"  Subsampling {original_size} to {MAX_SAMPLES_FOR_METRICS} for metrics...")
        indices = np.random.choice(original_size, size=MAX_SAMPLES_FOR_METRICS, replace=False)
        objectives_for_metrics = objectives[indices]
    else:
        objectives_for_metrics = objectives

    # Traditional metrics
    logger.info("  Computing traditional metrics...")
    reference_point = np.array([1.1] * env.num_objectives)
    traditional = compute_all_traditional_metrics(objectives_for_metrics, reference_point)
    metrics.update(traditional)

    # Trajectory metrics (sample some trajectories)
    logger.info("  Computing trajectory metrics (sampling trajectories)...")
    trajectories = []
    num_traj_samples = min(100, eval_samples)
    for i in range(num_traj_samples):
        traj = hngfn.sample_trajectory(explore=False)
        if traj.is_terminal:
            # Convert to standard Trajectory format for metrics
            from src.models.gflownet import Trajectory
            traj_converted = Trajectory(
                states=traj.states,
                actions=traj.actions,
                log_probs=traj.log_probs,
                is_terminal=traj.is_terminal
            )
            trajectories.append(traj_converted)

    if len(trajectories) > 0:
        metrics['tds'] = trajectory_diversity_score(trajectories)
        metrics['mpd'] = multi_path_diversity(trajectories)
    else:
        metrics['tds'] = 0.0
        metrics['mpd'] = 0.0

    # Spatial metrics
    logger.info("  Computing spatial metrics...")
    metrics['mce'], metrics['num_modes'] = mode_coverage_entropy(objectives_for_metrics)
    metrics['pmd'] = pairwise_minimum_distance(objectives_for_metrics)
    metrics['pfs'] = pareto_front_smoothness(objectives_for_metrics)
    metrics['num_unique_solutions'] = len(np.unique(objectives, axis=0))

    # Objective metrics (simplified PAS)
    logger.info("  Computing PAS...")
    if len(objectives_for_metrics) > 10:
        dists = pdist(objectives_for_metrics, metric='euclidean')
        metrics['pas'] = float(np.mean(dists))
    else:
        metrics['pas'] = 0.0

    # Dynamics metrics
    logger.info("  Computing dynamics metrics...")
    if len(trajectories) > 0:
        metrics['rbd'] = replay_buffer_diversity(trajectories, metric='trajectory_distance')
    else:
        metrics['rbd'] = 0.0

    # Flow metrics
    logger.info("  Computing flow metrics...")
    state_visits = {}
    for traj in trajectories:
        for state in traj.states:
            state_key = to_hashable(state)
            state_visits[state_key] = state_visits.get(state_key, 0) + 1
    if state_visits:
        metrics['fci'] = flow_concentration_index(state_visits)
    else:
        metrics['fci'] = 0.0

    # Composite metrics
    logger.info("  Computing composite metrics...")
    qds_results = quality_diversity_score(objectives_for_metrics, reference_point, alpha=0.5)
    metrics['qds'] = qds_results['qds']

    der_results = diversity_efficiency_ratio(
        objectives_for_metrics,
        training_time=training_time,
        num_parameters=num_params
    )
    metrics['der'] = der_results['der']

    # Metadata
    metrics['num_parameters'] = num_params
    metrics['training_time'] = training_time
    metrics['final_loss'] = training_history['loss'][-1] if training_history.get('loss') else 0.0
    metrics['algorithm'] = 'hngfn'
    metrics['num_samples'] = original_size
    metrics['num_samples_for_metrics'] = len(objectives_for_metrics)
    metrics['seed'] = seed
    metrics['hidden_dim'] = hidden_dim
    metrics['num_layers'] = num_layers
    metrics['z_hidden_dim'] = z_hidden_dim
    metrics['z_num_layers'] = z_num_layers

    # Save results
    logger.info(f"Saving results to {exp_dir}...")
    hngfn.save(str(exp_dir / 'checkpoint.pt'))

    np.save(exp_dir / 'objectives.npy', objectives)

    # Get Pareto front
    pareto_front = hngfn.get_pareto_front()
    np.save(exp_dir / 'pareto_front.npy', pareto_front)
    metrics['pareto_size'] = len(pareto_front)

    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    logger.info(
        f"HN-GFN complete: {len(objectives)} solutions, "
        f"{len(pareto_front)} Pareto optimal, HV={metrics.get('hypervolume', 0):.4f}"
    )

    return metrics


def aggregate_results(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate results across seeds into DataFrame."""
    if not results_list:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(results_list)

    # Compute summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].agg(['mean', 'std', 'min', 'max'])

    logger.info("\nSummary Statistics:")
    logger.info(summary.to_string())

    return df


def main():
    # Task-specific iteration defaults (from factorial experiments)
    TASK_DEFAULTS = {
        'hypergrid': {'iterations': 4000, 'batch_size': 128},
        'ngrams': {'iterations': 8000, 'batch_size': 128},
        'molecules': {'iterations': 10000, 'batch_size': 128},
        'sequences': {'iterations': 20000, 'batch_size': 128}
    }

    parser = argparse.ArgumentParser(
        description='Run baseline comparison',
        epilog="""
Task-specific defaults (from validated factorial experiments):
  hypergrid:  4,000 iterations, batch_size=128
  ngrams:     8,000 iterations, batch_size=128
  molecules: 10,000 iterations, batch_size=128
  sequences: 20,000 iterations, batch_size=128

Examples:
  # Use task-specific defaults (4000 iterations for hypergrid)
  python scripts/baselines/run_baseline_comparison.py \\
      --task hypergrid --algorithms mogfn_pc,hngfn,random,nsga2 --seeds 42,153,264 \\
      --output_dir results/baselines/hypergrid

  # Compare MOGFN-PC and HN-GFN
  python scripts/baselines/run_baseline_comparison.py \\
      --task hypergrid --algorithms mogfn_pc,hngfn --seeds 42,153,264,375,486 \\
      --output_dir results/baselines/mogfn_vs_hngfn

  # Override with custom iterations
  python scripts/baselines/run_baseline_comparison.py \\
      --task hypergrid --algorithms random --seeds 42 --num_iterations 1000 \\
      --output_dir results/baselines/test
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--task', type=str, default='hypergrid',
                       choices=['hypergrid', 'ngrams', 'molecules', 'sequences'],
                       help='Task/environment to use')
    parser.add_argument('--algorithms', type=str, default='random,nsga2',
                       help='Comma-separated list of algorithms to run')
    parser.add_argument('--seeds', type=str, default='42',
                       help='Comma-separated list of random seeds')
    parser.add_argument('--num_iterations', type=int, default=None,
                       help='Number of training iterations/generations (default: task-specific)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for Random/MOGFN-PC sampling (default: task-specific, typically 128)')
    parser.add_argument('--pop_size', type=int, default=100,
                       help='Population size for NSGA-II')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')

    # Environment-specific args
    parser.add_argument('--height', type=int, default=8,
                       help='Grid height (HyperGrid only)')
    parser.add_argument('--num_objectives', type=int, default=2,
                       help='Number of objectives')
    parser.add_argument('--reward_config', type=str, default='corners',
                       help='Reward configuration (HyperGrid only)')

    # MOGFN-PC and HN-GFN specific args
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for policy networks in MOGFN-PC/HN-GFN (default: 128)')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of layers for policy networks in MOGFN-PC/HN-GFN (default: 4)')
    parser.add_argument('--z_hidden_dim', type=int, default=64,
                       help='Hidden dimension for Z hypernetwork in HN-GFN (default: 64)')
    parser.add_argument('--z_num_layers', type=int, default=3,
                       help='Number of layers for Z hypernetwork in HN-GFN (default: 3)')
    parser.add_argument('--eval_samples', type=int, default=1000,
                       help='Number of evaluation samples for MOGFN-PC/HN-GFN (default: 1000)')

    args = parser.parse_args()

    # Apply task-specific defaults if not specified
    if args.num_iterations is None:
        args.num_iterations = TASK_DEFAULTS[args.task]['iterations']
        logger.info(f"Using task-specific default: {args.num_iterations} iterations for {args.task}")

    if args.batch_size is None:
        args.batch_size = TASK_DEFAULTS[args.task]['batch_size']
        logger.info(f"Using task-specific default: batch_size={args.batch_size} for {args.task}")

    # Parse arguments
    algorithms = args.algorithms.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Baseline Comparison Experiment")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Algorithms: {algorithms}")
    logger.info(f"  Seeds: {seeds}")
    logger.info(f"  Iterations: {args.num_iterations}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Output: {output_dir}")

    # Create environment
    env = create_environment(
        args.task,
        height=args.height,
        num_objectives=args.num_objectives,
        reward_config=args.reward_config
    )
    logger.info(f"Environment: {env.__class__.__name__} ({env.num_objectives} objectives)")

    # Run experiments
    all_results = []

    for algorithm in algorithms:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running {algorithm.upper()} baseline")
        logger.info(f"{'='*70}")

        algo_results = []

        for seed in seeds:
            try:
                if algorithm == 'random':
                    metrics = run_random_baseline(
                        env,
                        seed=seed,
                        num_iterations=args.num_iterations,
                        batch_size=args.batch_size,
                        output_dir=output_dir
                    )
                elif algorithm == 'nsga2':
                    # Adjust log interval based on num_iterations for better feedback
                    log_interval = max(1, args.num_iterations // 20)  # Log ~20 times
                    logger.info(f"NSGA-II will log every {log_interval} generations")

                    metrics = run_nsga2_baseline(
                        env,
                        seed=seed,
                        num_iterations=args.num_iterations,
                        pop_size=args.pop_size,
                        output_dir=output_dir,
                        log_interval=log_interval
                    )
                elif algorithm == 'mogfn_pc':
                    metrics = run_mogfn_pc_baseline(
                        env,
                        seed=seed,
                        num_iterations=args.num_iterations,
                        batch_size=args.batch_size,
                        output_dir=output_dir,
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers,
                        eval_samples=args.eval_samples
                    )
                elif algorithm == 'hngfn':
                    metrics = run_hngfn_baseline(
                        env,
                        seed=seed,
                        num_iterations=args.num_iterations,
                        batch_size=args.batch_size,
                        output_dir=output_dir,
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers,
                        z_hidden_dim=args.z_hidden_dim,
                        z_num_layers=args.z_num_layers,
                        eval_samples=args.eval_samples
                    )
                else:
                    logger.warning(f"Unknown algorithm: {algorithm}")
                    continue

                if metrics:
                    algo_results.append(metrics)
                    all_results.append(metrics)

            except Exception as e:
                logger.error(f"Error running {algorithm} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()

        # Aggregate results for this algorithm
        if algo_results:
            df = aggregate_results(algo_results)
            df.to_csv(output_dir / f'{algorithm}_results.csv', index=False)

    # Create overall summary
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(output_dir / 'all_results.csv', index=False)

        # Summary by algorithm
        summary = df_all.groupby('algorithm').agg({
            'hypervolume': ['mean', 'std'],
            'spacing': ['mean', 'std'],
            'mce': ['mean', 'std'],
            'pmd': ['mean', 'std'],
            'qds': ['mean', 'std'],
            'training_time': ['mean', 'std']
        })

        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY BY ALGORITHM")
        logger.info("="*70)
        logger.info(summary.to_string())

        summary.to_csv(output_dir / 'summary_by_algorithm.csv')

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Done!")


if __name__ == '__main__':
    main()