"""
Reproduce baseline MOGFN-PC results on HyperGrid task.

This script tests your MOGFN-PC implementation on the HyperGrid environment
to verify it matches the results from Jain et al. (2023).

Usage:
    Run directly:
    python scripts/baseline/reproduce_baseline.py
"""

import sys
from pathlib import Path
import argparse
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt

# Dynamic import for tqdm to avoid a hard dependency in environments where it's not installed.
# Falls back to a no-op iterator (returns the input iterable) if tqdm is unavailable.
_tqdm = None
try:
    _tqdm = importlib.import_module('tqdm').tqdm
except Exception:
    def _no_tqdm(iterable, *args, **kwargs):
        return iterable
    _tqdm = _no_tqdm

tqdm = _tqdm

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from models.mogfn_pc import MOGFN_PC, PreferenceSampler, MOGFNTrainer, MOGFNSampler
from environments.hypergrid import HyperGrid


def compute_hypervolume(objectives: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume indicator.
    
    Simple 2D implementation. For more objectives, use pymoo.
    """
    if objectives.shape[1] != 2:
        print("Hypervolume computation only implemented for 2 objectives")
        return 0.0
    
    # Sort by first objective
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_indices]
    
    # Compute hypervolume
    hv = 0.0
    prev_x = reference_point[0]
    
    for i in range(len(sorted_obj)):
        x, y = sorted_obj[i]
        if x >= reference_point[0] or y >= reference_point[1]:
            continue
        
        width = prev_x - x
        height = reference_point[1] - y
        hv += width * height
        prev_x = x
    
    return hv


def compute_diversity_metrics(objectives: np.ndarray) -> dict:
    """Compute diversity metrics for generated solutions."""
    from scipy.spatial.distance import pdist
    
    metrics = {}
    
    # Average pairwise distance
    if len(objectives) > 1:
        distances = pdist(objectives, metric='euclidean')
        metrics['avg_pairwise_distance'] = np.mean(distances)
        metrics['std_pairwise_distance'] = np.std(distances)
    else:
        metrics['avg_pairwise_distance'] = 0.0
        metrics['std_pairwise_distance'] = 0.0
    
    # Spread (range in each objective)
    metrics['spread_obj1'] = np.max(objectives[:, 0]) - np.min(objectives[:, 0])
    metrics['spread_obj2'] = np.max(objectives[:, 1]) - np.min(objectives[:, 1])
    
    return metrics


def evaluate_mogfn(mogfn: MOGFN_PC,
                env: HyperGrid,
                preference_sampler: PreferenceSampler,
                num_samples: int = 1000,
                device: str = 'cpu') -> dict:
    """
    Comprehensive evaluation of MOGFN.
    
    Args:
        mogfn: Trained MOGFN model
        env: HyperGrid environment
        preference_sampler: Preference sampler
        num_samples: Number of samples to generate
        device: Device to use
    
    Returns:
        results: Dictionary of evaluation metrics
    """
    mogfn.eval()
    sampler = MOGFNSampler(mogfn, env, preference_sampler)
    
    all_objectives = []
    all_preferences = []
    all_states = []
    
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        preferences = preference_sampler.sample(num_samples)
        
        for i in tqdm(range(num_samples)):
            traj = sampler.sample_trajectory(preferences[i], explore=False)
            all_objectives.append(traj.reward.cpu())
            all_preferences.append(preferences[i].cpu())
            all_states.append(traj.states[-1].cpu())  # Terminal state
    
    all_objectives = torch.stack(all_objectives).numpy()
    all_preferences = torch.stack(all_preferences).numpy()
    all_states = torch.stack(all_states).numpy()
    
    # Compute metrics
    results = {}
    
    # Hypervolume (reference point slightly worse than worst possible)
    reference_point = np.array([1.1, 1.1])
    results['hypervolume'] = compute_hypervolume(all_objectives, reference_point)
    
    # Diversity metrics
    diversity = compute_diversity_metrics(all_objectives)
    results.update(diversity)
    
    # Coverage (number of unique states)
    unique_states = np.unique(all_states, axis=0)
    results['unique_states'] = len(unique_states)
    results['state_coverage'] = len(unique_states) / (env.height ** 2)
    
    # Store data for visualization
    results['objectives'] = all_objectives
    results['preferences'] = all_preferences
    results['states'] = all_states
    
    return results


def plot_results(results: dict, 
                save_dir: Path,
                env: HyperGrid):
    """Plot and save evaluation results."""
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    objectives = results['objectives']
    
    # 1. Objective space scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(objectives[:, 0], objectives[:, 1], 
            alpha=0.5, s=20, c='blue', edgecolors='none')
    plt.xlabel('Objective 1', fontsize=12)
    plt.ylabel('Objective 2', fontsize=12)
    plt.title('MOGFN-PC: Generated Solutions in Objective Space', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'objective_space.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. State space heatmap
    states = results['states']
    heatmap = np.zeros((env.height, env.height))
    for state in states:
        x, y = int(state[0]), int(state[1])
        heatmap[y, x] += 1
    
    plt.figure(figsize=(8, 7))
    plt.imshow(heatmap, cmap='hot', origin='lower', interpolation='nearest')
    plt.colorbar(label='Visit Count')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('MOGFN-PC: State Visitation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'state_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Training curves (if available)
    # This would be added during training
    
    print(f"Plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Reproduce MOGFN-PC baseline on HyperGrid')
    parser.add_argument('--height', type=int, default=8, help='Grid height')
    parser.add_argument('--num_objectives', type=int, default=2, help='Number of objectives')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=10000, help='Training iterations')
    parser.add_argument('--alpha', type=float, default=1.5, help='Dirichlet alpha')
    parser.add_argument('--beta', type=float, default=1.0, help='Reward exponent')
    parser.add_argument('--encoding', type=str, default='vanilla', 
                    choices=['vanilla', 'thermometer'], help='Preference encoding')
    parser.add_argument('--conditioning', type=str, default='concat',
                    choices=['concat', 'film'], help='Conditioning type')
    parser.add_argument('--eval_every', type=int, default=1000, help='Evaluation frequency')
    parser.add_argument('--eval_samples', type=int, default=1000, help='Samples for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='results/baseline', help='Save directory')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MOGFN-PC Baseline Reproduction on HyperGrid")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Grid size: {args.height} x {args.height}")
    print(f"  Num objectives: {args.num_objectives}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num iterations: {args.num_iterations}")
    print(f"  Dirichlet alpha: {args.alpha}")
    print(f"  Reward beta: {args.beta}")
    print(f"  Preference encoding: {args.encoding}")
    print(f"  Conditioning type: {args.conditioning}")
    print(f"  Device: {args.device}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Create environment
    print("Creating HyperGrid environment...")
    env = HyperGrid(
        height=args.height,
        num_objectives=args.num_objectives,
        reward_config='corners'
    )
    print(f"  State dim: {env.state_dim}")
    print(f"  Num actions: {env.num_actions}")
    print(f"  Num objectives: {env.num_objectives}")
    
    # Visualize objectives
    print("\nVisualizing objective landscape...")
    env.visualize_objectives(save_path=save_dir / 'objective_landscape.png')
    
    # Create MOGFN-PC
    print("\nCreating MOGFN-PC model...")
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=args.hidden_dim,
        num_actions=env.num_actions,
        num_layers=args.num_layers,
        preference_encoding=args.encoding,
        conditioning_type=args.conditioning
    ).to(args.device)
    
    num_params = sum(p.numel() for p in mogfn.parameters())
    print(f"  Total parameters: {num_params:,}")
    
    # Create preference sampler
    print("\nCreating preference sampler...")
    preference_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution='dirichlet',
        alpha=args.alpha
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(mogfn.parameters(), lr=args.lr)
    
    # Create trainer
    trainer = MOGFNTrainer(
        mogfn=mogfn,
        env=env,
        preference_sampler=preference_sampler,
        optimizer=optimizer,
        beta=args.beta
    )
    
    # Training loop with periodic evaluation
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    training_history = {
        'iteration': [],
        'loss': [],
        'log_Z': [],
        'hypervolume': [],
        'diversity': []
    }
    
    for iteration in range(args.num_iterations):
        # Training step
        metrics = trainer.train_step(batch_size=args.batch_size)
        
        # Log
        if iteration % 100 == 0:
            print(f"Iter {iteration:5d}/{args.num_iterations} - "
                f"Loss: {metrics['loss']:.4f}, "
                f"log Z: {metrics['log_Z']:.4f}")
        
        # Periodic evaluation
        if iteration % args.eval_every == 0 or iteration == args.num_iterations - 1:
            print(f"\nEvaluation at iteration {iteration}...")
            eval_results = evaluate_mogfn(
                mogfn, env, preference_sampler, 
                num_samples=args.eval_samples,
                device=args.device
            )
            
            print(f"  Hypervolume: {eval_results['hypervolume']:.4f}")
            print(f"  Avg pairwise distance: {eval_results['avg_pairwise_distance']:.4f}")
            print(f"  Unique states: {eval_results['unique_states']}")
            print(f"  State coverage: {eval_results['state_coverage']:.2%}")
            
            # Track metrics
            training_history['iteration'].append(iteration)
            training_history['loss'].append(metrics['loss'])
            training_history['log_Z'].append(metrics['log_Z'])
            training_history['hypervolume'].append(eval_results['hypervolume'])
            training_history['diversity'].append(eval_results['avg_pairwise_distance'])
            
            # Save checkpoint
            checkpoint_path = save_dir / f'checkpoint_iter{iteration}.pt'
            torch.save({
                'iteration': iteration,
                'model_state_dict': mogfn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': eval_results,
            }, checkpoint_path)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    final_results = evaluate_mogfn(
        mogfn, env, preference_sampler,
        num_samples=5000,  # More samples for final eval
        device=args.device
    )
    
    print(f"\nFinal Results:")
    print(f"  Hypervolume: {final_results['hypervolume']:.4f}")
    print(f"  Avg pairwise distance: {final_results['avg_pairwise_distance']:.4f}")
    print(f"  Spread (Obj 1): {final_results['spread_obj1']:.4f}")
    print(f"  Spread (Obj 2): {final_results['spread_obj2']:.4f}")
    print(f"  Unique states: {final_results['unique_states']}")
    print(f"  State coverage: {final_results['state_coverage']:.2%}")