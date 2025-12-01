#!/usr/bin/env python3
"""
Visualize discovered modes on HyperGrid from experimental results.

This script loads objectives from ablation or factorial experiment results
and visualizes them on the HyperGrid to show the distribution of discovered modes.

For HyperGrid environments, the objectives correspond to rewards at different
corners/regions, allowing us to infer approximate grid positions and visualize
which modes the model discovered.

Usage:
    # Visualize a specific experiment
    python scripts/validation/visualize_hypergrid_modes.py \
        --experiment results/ablations/capacity/small_concat_seed42

    # Compare multiple experiments
    python scripts/validation/visualize_hypergrid_modes.py \
        --experiments results/ablations/capacity/small_concat_seed42 \
                     results/ablations/capacity/large_concat_seed42 \
        --output_dir results/validation/mode_visualizations

    # Visualize all experiments from an ablation study
    python scripts/validation/visualize_hypergrid_modes.py \
        --ablation_dir results/ablations/capacity \
        --output_dir results/validation/capacity_modes

    # Visualize factorial results
    python scripts/validation/visualize_hypergrid_modes.py \
        --ablation_dir results/factorials/hypergrid \
        --output_dir results/validation/factorial_modes

    # Visualize baseline comparisons
    python scripts/validation/visualize_hypergrid_modes.py \
        --ablation_dir results/baselines/hypergrid \
        --output_dir results/validation/baseline_modes

    # Compare specific baselines
    python scripts/validation/visualize_hypergrid_modes.py \
        --experiments results/baselines/hypergrid/mogfn_pc_seed42 \
                     results/baselines/hypergrid/hngfn_seed42 \
                     results/baselines/hypergrid/nsga2_seed42 \
                     results/baselines/hypergrid/random_seed42 \
        --output_dir results/validation/baseline_comparison

    # Aggregate all seeds for each baseline algorithm (creates only 4 plots, not 20)
    python scripts/validation/visualize_hypergrid_modes.py \
        --ablation_dir results/baselines/hypergrid \
        --aggregate_seeds \
        --output_dir results/validation/baseline_modes_aggregated
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments.hypergrid import HyperGrid


def extract_algorithm_name(exp_name: str) -> str:
    """
    Extract algorithm/configuration name from experiment directory name.

    Examples:
        'mogfn_pc_seed42' -> 'mogfn_pc'
        'hngfn_seed153' -> 'hngfn'
        'small_concat_seed42' -> 'small_concat'

    Args:
        exp_name: Experiment directory name

    Returns:
        Algorithm/configuration name without seed
    """
    # Remove seed suffix (e.g., '_seed42', '_seed153')
    import re
    return re.sub(r'_seed\d+$', '', exp_name)


def aggregate_experiments_by_algorithm(exp_dirs: List[Path]) -> Dict[str, List[Path]]:
    """
    Group experiment directories by algorithm/configuration name.

    Args:
        exp_dirs: List of experiment directories

    Returns:
        Dictionary mapping algorithm name to list of experiment directories
    """
    grouped = {}
    for exp_dir in exp_dirs:
        algo_name = extract_algorithm_name(exp_dir.name)
        if algo_name not in grouped:
            grouped[algo_name] = []
        grouped[algo_name].append(exp_dir)

    return grouped


def load_experiment_data(exp_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Load objectives, preferences, and metadata from experiment directory.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        objectives: Array of shape (N, num_objectives)
        preferences: Array of shape (N, num_objectives) or None if not available
        metadata: Dictionary with experiment configuration
    """
    objectives = np.load(exp_dir / 'objectives.npy')

    # Preferences may not exist for baseline methods (e.g., NSGA-II, random)
    pref_path = exp_dir / 'preferences.npy'
    preferences = np.load(pref_path) if pref_path.exists() else None

    # Load metadata
    with open(exp_dir / 'metrics.json', 'r') as f:
        metadata = json.load(f)

    return objectives, preferences, metadata


def load_aggregated_data(exp_dirs: List[Path]) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Load and aggregate data from multiple experiment directories (e.g., all seeds).

    Args:
        exp_dirs: List of experiment directories to aggregate

    Returns:
        objectives: Concatenated array from all experiments
        preferences: Concatenated array from all experiments (or None)
        metadata: Metadata from first experiment (assumed consistent)
    """
    all_objectives = []
    all_preferences = []
    has_preferences = False
    metadata = None

    for exp_dir in exp_dirs:
        objectives, preferences, meta = load_experiment_data(exp_dir)
        all_objectives.append(objectives)

        if preferences is not None:
            all_preferences.append(preferences)
            has_preferences = True

        if metadata is None:
            metadata = meta

    # Concatenate all objectives
    combined_objectives = np.concatenate(all_objectives, axis=0)

    # Concatenate preferences if available
    combined_preferences = None
    if has_preferences:
        combined_preferences = np.concatenate(all_preferences, axis=0)

    # Update metadata with aggregation info
    metadata['aggregated_seeds'] = [extract_algorithm_name(d.name).split('_')[-1] for d in exp_dirs]
    metadata['num_seeds'] = len(exp_dirs)

    return combined_objectives, combined_preferences, metadata


def infer_grid_positions_2obj(objectives: np.ndarray,
                               grid_height: int = 8,
                               use_exact_inverse: bool = True) -> np.ndarray:
    """
    Infer grid positions from 2-objective rewards.

    For the standard 2-objective HyperGrid with 'corners' config:
    - Objective 1: High at top-right corner (7,7)
    - Objective 2: High at top-left corner (0,7)

    Reward functions are: R_i = exp(-0.5 * dist_to_corner_i)

    Args:
        objectives: Array of shape (N, 2) with [obj1, obj2] values
        grid_height: Grid size (default: 8)
        use_exact_inverse: If True, use exact geometric inverse (recommended)

    Returns:
        positions: Array of shape (N, 2) with [x, y] positions
    """
    positions = []
    corner1 = np.array([grid_height - 1, grid_height - 1])  # Top-right (7,7)
    corner2 = np.array([0, grid_height - 1])  # Top-left (0,7)

    for obj1, obj2 in objectives:
        if use_exact_inverse:
            # Exact inverse using distance formula
            # R1 = exp(-0.5 * dist1) => dist1 = -2 * ln(R1)
            # R2 = exp(-0.5 * dist2) => dist2 = -2 * ln(R2)

            # Clamp to avoid log(0)
            obj1_safe = max(obj1, 1e-10)
            obj2_safe = max(obj2, 1e-10)

            dist1 = np.sqrt(-2 * np.log(obj1_safe))
            dist2 = np.sqrt(-2 * np.log(obj2_safe))

            # We have:
            # dist((x,y), (7,7)) = dist1
            # dist((x,y), (0,7)) = dist2
            #
            # (x-7)^2 + (y-7)^2 = dist1^2
            # (x-0)^2 + (y-7)^2 = dist2^2
            #
            # Subtracting: x^2 - 14x + 49 - x^2 = dist1^2 - dist2^2
            # => -14x + 49 = dist1^2 - dist2^2
            # => x = (49 - (dist1^2 - dist2^2)) / 14

            x = (49 - (dist1**2 - dist2**2)) / 14
            x = np.clip(x, 0, grid_height - 1)

            # Solve for y using first equation
            # (x-7)^2 + (y-7)^2 = dist1^2
            # (y-7)^2 = dist1^2 - (x-7)^2
            # y = 7 ± sqrt(dist1^2 - (x-7)^2)

            discriminant = dist1**2 - (x - 7)**2
            if discriminant >= 0:
                # Choose the solution closer to top (y=7)
                y = 7 - np.sqrt(discriminant)
                y = np.clip(y, 0, grid_height - 1)
            else:
                # If discriminant is negative (numerical error), estimate from both corners
                y = 7 - dist1 / np.sqrt(2)
                y = np.clip(y, 0, grid_height - 1)

        else:
            # Fallback: weighted interpolation (old method)
            total = obj1 + obj2 + 1e-8
            w1 = obj1 / total
            x = w1 * (grid_height - 1)

            y_from_obj1 = (grid_height - 1) - np.sqrt(-2 * np.log(max(obj1, 1e-8))) / np.sqrt(2)
            y_from_obj2 = (grid_height - 1) - np.sqrt(-2 * np.log(max(obj2, 1e-8))) / np.sqrt(2)
            y = max(y_from_obj1, y_from_obj2)

            x = np.clip(x, 0, grid_height - 1)
            y = np.clip(y, 0, grid_height - 1)

        positions.append([x, y])

    return np.array(positions)


def visualize_modes_grid(objectives: np.ndarray,
                         preferences: Optional[np.ndarray] = None,
                         title: str = "Discovered Modes on HyperGrid",
                         grid_height: int = 8,
                         save_path: Optional[Path] = None) -> plt.Figure:
    """
    Visualize discovered modes on the HyperGrid.

    Args:
        objectives: Array of shape (N, num_objectives)
        preferences: Optional array of shape (N, num_objectives)
        title: Plot title
        grid_height: Grid size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    num_objectives = objectives.shape[1]

    if num_objectives == 2:
        # Create figure with two subplots: objective space + grid heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Objective space scatter
        ax_obj = axes[0]
        scatter = ax_obj.scatter(objectives[:, 0], objectives[:, 1],
                                alpha=0.5, s=20, c='blue', edgecolors='black', linewidth=0.3)
        ax_obj.set_xlabel('Objective 1 (Top-Right Reward)', fontsize=12, fontweight='bold')
        ax_obj.set_ylabel('Objective 2 (Top-Left Reward)', fontsize=12, fontweight='bold')
        ax_obj.set_title('(A) Objective Space', fontsize=13, fontweight='bold')
        ax_obj.grid(alpha=0.3)
        ax_obj.set_xlim(-0.05, 1.05)
        ax_obj.set_ylim(-0.05, 1.05)

        # Add corner labels
        ax_obj.text(0.95, 0.05, 'Top-Right\nCorner', ha='right', va='bottom',
                   fontsize=9, style='italic', color='red',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_obj.text(0.05, 0.95, 'Top-Left\nCorner', ha='left', va='top',
                   fontsize=9, style='italic', color='green',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Right: Grid heatmap showing mode density
        ax_grid = axes[1]

        # Infer approximate positions
        positions = infer_grid_positions_2obj(objectives, grid_height)

        # Create 2D histogram (heatmap)
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=grid_height,
            range=[[0, grid_height], [0, grid_height]]
        )

        # Plot heatmap
        im = ax_grid.imshow(heatmap.T, origin='lower', cmap='YlOrRd',
                           aspect='equal', interpolation='nearest',
                           extent=[0, grid_height, 0, grid_height])

        # Add gridlines
        for i in range(grid_height + 1):
            ax_grid.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
            ax_grid.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

        # Mark corners
        corner_size = 300
        ax_grid.scatter([grid_height-0.5], [grid_height-0.5],
                       s=corner_size, c='red', marker='*',
                       edgecolors='darkred', linewidth=2,
                       label='Top-Right (Obj 1)', zorder=10)
        ax_grid.scatter([0.5], [grid_height-0.5],
                       s=corner_size, c='green', marker='*',
                       edgecolors='darkgreen', linewidth=2,
                       label='Top-Left (Obj 2)', zorder=10)

        ax_grid.set_xlabel('x (right →)', fontsize=12, fontweight='bold')
        ax_grid.set_ylabel('y (up →)', fontsize=12, fontweight='bold')
        ax_grid.set_title('(B) Mode Density on Grid', fontsize=13, fontweight='bold')
        ax_grid.legend(loc='lower left', fontsize=9)
        ax_grid.set_xlim(0, grid_height)
        ax_grid.set_ylim(0, grid_height)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_grid)
        cbar.set_label('Sample Count', fontsize=11, fontweight='bold')

        # Overall title
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

        # Add statistics
        num_unique = len(np.unique(objectives, axis=0))
        stats_text = f'Total samples: {len(objectives)} | Unique modes: {num_unique}'
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    else:
        # For 3+ objectives, use parallel coordinates in objective space
        fig, ax = plt.subplots(figsize=(12, 8))

        for i in range(min(len(objectives), 500)):  # Limit for visibility
            ax.plot(range(num_objectives), objectives[i],
                   alpha=0.3, linewidth=0.5, color='blue')

        ax.set_xlabel('Objective Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(num_objectives))
        ax.set_xticklabels([f'Obj {i+1}' for i in range(num_objectives)])
        ax.grid(alpha=0.3)

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")

    return fig


def visualize_reward_landscape(grid_height: int = 8,
                               num_objectives: int = 2,
                               reward_config: str = 'corners',
                               save_path: Optional[Path] = None) -> plt.Figure:
    """
    Visualize the reward landscape for each objective.

    Args:
        grid_height: Grid size
        num_objectives: Number of objectives
        reward_config: Reward configuration ('corners' or 'modes')
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    env = HyperGrid(height=grid_height,
                   num_objectives=num_objectives,
                   reward_config=reward_config)

    # Create grid of positions
    x_coords = np.arange(grid_height)
    y_coords = np.arange(grid_height)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Compute rewards for each objective
    fig, axes = plt.subplots(1, num_objectives, figsize=(6*num_objectives, 5))

    if num_objectives == 1:
        axes = [axes]

    for i, (ax, reward_fn) in enumerate(zip(axes, env.reward_functions)):
        Z = np.zeros_like(X, dtype=float)

        for ix in range(grid_height):
            for iy in range(grid_height):
                Z[iy, ix] = reward_fn(ix, iy)

        im = ax.imshow(Z, origin='lower', cmap='viridis', aspect='equal',
                      extent=[0, grid_height, 0, grid_height])
        ax.set_xlabel('x (right →)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (up →)', fontsize=11, fontweight='bold')
        ax.set_title(f'Objective {i+1} Reward Landscape',
                    fontsize=12, fontweight='bold')

        # Add gridlines
        for j in range(grid_height + 1):
            ax.axhline(j, color='white', linewidth=0.5, alpha=0.3)
            ax.axvline(j, color='white', linewidth=0.5, alpha=0.3)

        plt.colorbar(im, ax=ax, label='Reward Value')

    fig.suptitle(f'HyperGrid Reward Landscapes ({reward_config} config)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved reward landscape: {save_path.name}")

    return fig


def compare_experiments(exp_dirs: List[Path],
                       grid_height: int = 8,
                       save_path: Optional[Path] = None) -> plt.Figure:
    """
    Compare mode distributions across multiple experiments.

    Args:
        exp_dirs: List of experiment directories
        grid_height: Grid size
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    n_exps = len(exp_dirs)
    ncols = min(3, n_exps)
    nrows = (n_exps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_exps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 else axes

    for idx, exp_dir in enumerate(exp_dirs):
        ax = axes[idx]

        # Load data
        objectives, _, metadata = load_experiment_data(exp_dir)

        # Infer positions
        if objectives.shape[1] == 2:
            positions = infer_grid_positions_2obj(objectives, grid_height)

            # Create heatmap
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=grid_height,
                range=[[0, grid_height], [0, grid_height]]
            )

            im = ax.imshow(heatmap.T, origin='lower', cmap='YlOrRd',
                          aspect='equal', interpolation='nearest',
                          extent=[0, grid_height, 0, grid_height])

            # Add gridlines
            for i in range(grid_height + 1):
                ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

            # Mark corners
            ax.scatter([grid_height-0.5], [grid_height-0.5],
                      s=200, c='red', marker='*',
                      edgecolors='darkred', linewidth=1.5, zorder=10)
            ax.scatter([0.5], [grid_height-0.5],
                      s=200, c='green', marker='*',
                      edgecolors='darkgreen', linewidth=1.5, zorder=10)

            # Title with experiment name
            exp_name = exp_dir.name
            # For ablations/factorials, show capacity & conditioning
            # For baselines, show algorithm name
            if 'algorithm' in metadata:
                algorithm = metadata.get('algorithm', 'Unknown')
                title = f'{exp_name}\n{algorithm}'
            else:
                capacity = metadata.get('capacity', '?')
                conditioning = metadata.get('conditioning', '?')
                title = f'{exp_name}\n{capacity} | {conditioning}'
            ax.set_title(title, fontsize=10, fontweight='bold')

            ax.set_xlabel('x', fontsize=9)
            ax.set_ylabel('y', fontsize=9)

            plt.colorbar(im, ax=ax, label='Count')

    # Hide unused subplots
    for idx in range(n_exps, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Mode Distribution Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved comparison: {save_path.name}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize HyperGrid modes from experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--experiment',
        type=Path,
        help='Path to single experiment directory'
    )

    parser.add_argument(
        '--experiments',
        type=Path,
        nargs='+',
        help='Paths to multiple experiment directories to compare'
    )

    parser.add_argument(
        '--ablation_dir',
        type=Path,
        help='Path to ablation/factorial directory (visualizes all experiments)'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/validation/mode_visualizations'),
        help='Output directory for visualizations'
    )

    parser.add_argument(
        '--grid_height',
        type=int,
        default=8,
        help='Grid height (default: 8)'
    )

    parser.add_argument(
        '--num_objectives',
        type=int,
        default=2,
        help='Number of objectives (default: 2)'
    )

    parser.add_argument(
        '--reward_config',
        type=str,
        default='corners',
        choices=['corners', 'modes'],
        help='Reward configuration (default: corners)'
    )

    parser.add_argument(
        '--show_landscape',
        action='store_true',
        help='Generate reward landscape visualization'
    )

    parser.add_argument(
        '--aggregate_seeds',
        action='store_true',
        help='Aggregate results across all seeds for each algorithm/configuration'
    )

    args = parser.parse_args()

    print("="*80)
    print("HYPERGRID MODE VISUALIZATION")
    print("="*80)
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate reward landscape if requested
    if args.show_landscape:
        print("\nGenerating reward landscape...")
        landscape_path = args.output_dir / 'reward_landscape.pdf'
        visualize_reward_landscape(
            grid_height=args.grid_height,
            num_objectives=args.num_objectives,
            reward_config=args.reward_config,
            save_path=landscape_path
        )

    # Collect experiment directories
    exp_dirs = []

    if args.experiment:
        exp_dirs = [args.experiment]

    elif args.experiments:
        exp_dirs = args.experiments

    elif args.ablation_dir:
        # Find all experiment subdirectories
        exp_dirs = [
            d for d in args.ablation_dir.iterdir()
            if d.is_dir() and (d / 'objectives.npy').exists()
        ]
        exp_dirs.sort()
        print(f"\nFound {len(exp_dirs)} experiments in {args.ablation_dir}")

    else:
        parser.error("Must specify --experiment, --experiments, or --ablation_dir")

    # Validate experiment directories
    valid_exp_dirs = []
    for exp_dir in exp_dirs:
        if not (exp_dir / 'objectives.npy').exists():
            print(f"  ⚠️  Skipping {exp_dir.name}: objectives.npy not found")
            continue
        if not (exp_dir / 'metrics.json').exists():
            print(f"  ⚠️  Skipping {exp_dir.name}: metrics.json not found")
            continue
        valid_exp_dirs.append(exp_dir)

    if not valid_exp_dirs:
        print("\n✗ No valid experiment directories found!")
        return

    # Aggregate by algorithm/configuration if requested
    if args.aggregate_seeds:
        grouped = aggregate_experiments_by_algorithm(valid_exp_dirs)
        print(f"\nAggregating {len(valid_exp_dirs)} experiments into {len(grouped)} groups:")
        for algo_name, dirs in grouped.items():
            print(f"  - {algo_name}: {len(dirs)} seeds")

        # Process each algorithm group
        print(f"\nVisualizing {len(grouped)} aggregated groups...")

        if len(grouped) == 1:
            # Single algorithm: detailed visualization
            algo_name, exp_dirs_group = list(grouped.items())[0]
            print(f"\n{'='*80}")
            print(f"Algorithm: {algo_name}")
            print(f"{'='*80}")

            objectives, preferences, metadata = load_aggregated_data(exp_dirs_group)

            print(f"  Objectives shape: {objectives.shape}")
            if preferences is not None:
                print(f"  Preferences shape: {preferences.shape}")
            else:
                print(f"  Preferences: None (baseline method)")
            print(f"  Aggregated seeds: {metadata['num_seeds']}")
            print(f"  Algorithm: {metadata.get('algorithm', 'N/A')}")

            # Generate visualization
            save_path = args.output_dir / f"{algo_name}_aggregated_modes.pdf"
            visualize_modes_grid(
                objectives=objectives,
                preferences=preferences,
                title=f"Discovered Modes: {algo_name} ({metadata['num_seeds']} seeds)",
                grid_height=args.grid_height,
                save_path=save_path
            )

            print(f"\n✓ Saved visualization: {save_path}")

        else:
            # Multiple algorithms: comparison grid
            print(f"\n{'='*80}")
            print(f"Comparing {len(grouped)} algorithms (aggregated)")
            print(f"{'='*80}")

            # Create aggregated comparison
            algo_names = sorted(grouped.keys())
            aggregated_dirs = []

            # For comparison, we need to create temporary combined data
            for algo_name in algo_names:
                exp_dirs_group = grouped[algo_name]
                objectives, preferences, metadata = load_aggregated_data(exp_dirs_group)

                # Create a temporary identifier for this aggregated group
                # We'll use this in the comparison function
                temp_dir = exp_dirs_group[0].parent / f"{algo_name}_aggregated"
                temp_dir.mkdir(exist_ok=True)

                # Save aggregated data
                np.save(temp_dir / 'objectives.npy', objectives)
                if preferences is not None:
                    np.save(temp_dir / 'preferences.npy', preferences)

                # Save metadata
                with open(temp_dir / 'metrics.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

                aggregated_dirs.append(temp_dir)

            # Generate comparison plot
            save_path = args.output_dir / "mode_comparison_aggregated.pdf"
            compare_experiments(
                exp_dirs=aggregated_dirs,
                grid_height=args.grid_height,
                save_path=save_path
            )

            # Also generate individual plots
            print("\nGenerating individual visualizations...")
            for algo_name in algo_names:
                exp_dirs_group = grouped[algo_name]
                objectives, preferences, metadata = load_aggregated_data(exp_dirs_group)

                save_path = args.output_dir / f"{algo_name}_aggregated_modes.pdf"
                visualize_modes_grid(
                    objectives=objectives,
                    preferences=preferences,
                    title=f"Discovered Modes: {algo_name} ({metadata['num_seeds']} seeds)",
                    grid_height=args.grid_height,
                    save_path=save_path
                )

        return

    # Standard mode: visualize each experiment separately
    print(f"\nVisualizing {len(valid_exp_dirs)} experiments...")

    # Single experiment: detailed visualization
    if len(valid_exp_dirs) == 1:
        exp_dir = valid_exp_dirs[0]
        print(f"\n{'='*80}")
        print(f"Experiment: {exp_dir.name}")
        print(f"{'='*80}")

        objectives, preferences, metadata = load_experiment_data(exp_dir)

        print(f"  Objectives shape: {objectives.shape}")
        if preferences is not None:
            print(f"  Preferences shape: {preferences.shape}")
        else:
            print(f"  Preferences: None (baseline method)")
        print(f"  Capacity: {metadata.get('capacity', 'N/A')}")
        print(f"  Conditioning: {metadata.get('conditioning', 'N/A')}")
        print(f"  Algorithm: {metadata.get('algorithm', 'N/A')}")
        print(f"  Seed: {metadata.get('seed', 'N/A')}")

        # Generate visualization
        save_path = args.output_dir / f"{exp_dir.name}_modes.pdf"
        visualize_modes_grid(
            objectives=objectives,
            preferences=preferences,
            title=f"Discovered Modes: {exp_dir.name}",
            grid_height=args.grid_height,
            save_path=save_path
        )

        print(f"\n✓ Saved visualization: {save_path}")

    # Multiple experiments: comparison grid
    else:
        print(f"\n{'='*80}")
        print(f"Comparing {len(valid_exp_dirs)} experiments")
        print(f"{'='*80}")

        # Generate comparison plot
        save_path = args.output_dir / "mode_comparison.pdf"
        compare_experiments(
            exp_dirs=valid_exp_dirs,
            grid_height=args.grid_height,
            save_path=save_path
        )

        # Also generate individual plots
        print("\nGenerating individual visualizations...")
        for exp_dir in valid_exp_dirs:
            objectives, preferences, _ = load_experiment_data(exp_dir)

            save_path = args.output_dir / f"{exp_dir.name}_modes.pdf"
            visualize_modes_grid(
                objectives=objectives,
                preferences=preferences,
                title=f"Discovered Modes: {exp_dir.name}",
                grid_height=args.grid_height,
                save_path=save_path
            )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nVisualization includes:")
    print("  - Objective space scatter plots")
    print("  - Mode density heatmaps on grid")
    print("  - Corner marker annotations")
    print("  - Sample count statistics")


if __name__ == '__main__':
    main()
