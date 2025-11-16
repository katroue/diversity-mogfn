#!/usr/bin/env python3
"""
Simple test for baseline algorithms (no pytest required).
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import RandomSampler
from src.environments.hypergrid import HyperGrid
from src.metrics.traditional import compute_all_traditional_metrics
from src.metrics.spatial import mode_coverage_entropy, pairwise_minimum_distance

# Check if pymoo is available
try:
    from src.models.baselines import NSGA2Adapter
    NSGA2_AVAILABLE = True
except ImportError:
    NSGA2_AVAILABLE = False
    print("WARNING: pymoo not installed, skipping NSGA-II tests")
    print("Install with: pip install pymoo")


def test_random_sampler():
    """Test Random Sampling baseline."""
    print("\n" + "="*70)
    print("Testing RandomSampler")
    print("="*70)

    env = HyperGrid(height=4, num_objectives=2)
    sampler = RandomSampler(env, max_steps=20, seed=42)

    print("✓ RandomSampler initialized")

    # Sample a single trajectory
    traj = sampler.sample_trajectory()
    assert len(traj.states) > 0
    assert len(traj.objectives) == 2
    print(f"✓ Sample trajectory: {len(traj.states)} states, objectives={traj.objectives}")

    # Train
    print("\nTraining RandomSampler (50 iterations × 20 samples)...")
    history = sampler.train(num_iterations=50, batch_size=20, log_interval=10)
    print(f"✓ Training complete: {len(sampler.objectives_history)} terminal solutions")

    # Get objectives
    objectives = sampler.get_all_objectives()
    print(f"✓ Objectives shape: {objectives.shape}")

    # Pareto front
    pareto_front = sampler.get_pareto_front()
    print(f"✓ Pareto front: {len(pareto_front)} solutions")

    # Compute metrics
    reference_point = np.array([1.1, 1.1])
    traditional = compute_all_traditional_metrics(objectives, reference_point)
    print(f"✓ Hypervolume: {traditional['hypervolume']:.4f}")
    print(f"✓ Spacing: {traditional['spacing']:.4f}")

    mce, num_modes = mode_coverage_entropy(objectives)
    pmd = pairwise_minimum_distance(objectives)
    print(f"✓ MCE: {mce:.4f}, Modes: {num_modes}, PMD: {pmd:.4f}")

    print("\n✅ RandomSampler tests PASSED")


def test_nsga2_adapter():
    """Test NSGA-II baseline."""
    if not NSGA2_AVAILABLE:
        print("\n⚠️  Skipping NSGA-II tests (pymoo not installed)")
        return

    print("\n" + "="*70)
    print("Testing NSGA2Adapter")
    print("="*70)

    env = HyperGrid(height=4, num_objectives=2)
    nsga2 = NSGA2Adapter(env, pop_size=30, max_steps=20, seed=42)

    print("✓ NSGA2Adapter initialized")

    # Train
    print("\nTraining NSGA-II (10 generations, pop=30)...")
    history = nsga2.train(num_iterations=10, log_interval=3)
    print(f"✓ Training complete")

    # Get objectives
    objectives = nsga2.get_all_objectives()
    print(f"✓ Objectives shape: {objectives.shape}")

    # Pareto front
    pareto_front = nsga2.get_pareto_front()
    print(f"✓ Pareto front: {len(pareto_front)} solutions")

    # Compute metrics
    reference_point = np.array([1.1, 1.1])
    traditional = compute_all_traditional_metrics(objectives, reference_point)
    print(f"✓ Hypervolume: {traditional['hypervolume']:.4f}")
    print(f"✓ Spacing: {traditional['spacing']:.4f}")

    mce, num_modes = mode_coverage_entropy(objectives)
    pmd = pairwise_minimum_distance(objectives)
    print(f"✓ MCE: {mce:.4f}, Modes: {num_modes}, PMD: {pmd:.4f}")

    print("\n✅ NSGA2Adapter tests PASSED")


def test_comparison():
    """Test comparing baselines."""
    if not NSGA2_AVAILABLE:
        print("\n⚠️  Skipping comparison test (pymoo not installed)")
        return

    print("\n" + "="*70)
    print("Testing Baseline Comparison")
    print("="*70)

    env = HyperGrid(height=5, num_objectives=2)
    seed = 42
    reference_point = np.array([1.1, 1.1])

    # Random baseline
    print("\nRandom Sampling...")
    random_sampler = RandomSampler(env, max_steps=20, seed=seed)
    random_sampler.train(num_iterations=100, batch_size=20)
    random_objectives = random_sampler.get_all_objectives()
    random_metrics = compute_all_traditional_metrics(random_objectives, reference_point)

    # NSGA-II baseline
    print("NSGA-II...")
    nsga2 = NSGA2Adapter(env, pop_size=50, max_steps=20, seed=seed)
    nsga2.train(num_iterations=20)
    nsga2_objectives = nsga2.get_all_objectives()
    nsga2_metrics = compute_all_traditional_metrics(nsga2_objectives, reference_point)

    # Compare
    print("\n" + "-"*70)
    print("COMPARISON RESULTS:")
    print("-"*70)
    print(f"{'Metric':<20} {'Random':<20} {'NSGA-II':<20}")
    print("-"*70)
    print(f"{'Hypervolume':<20} {random_metrics.get('hypervolume', 0):<20.4f} {nsga2_metrics.get('hypervolume', 0):<20.4f}")
    print(f"{'Spacing':<20} {random_metrics.get('spacing', 0):<20.4f} {nsga2_metrics.get('spacing', 0):<20.4f}")
    if 'gd' in random_metrics and 'gd' in nsga2_metrics:
        print(f"{'GD':<20} {random_metrics['gd']:<20.4f} {nsga2_metrics['gd']:<20.4f}")
    if 'igd' in random_metrics and 'igd' in nsga2_metrics:
        print(f"{'IGD':<20} {random_metrics['igd']:<20.4f} {nsga2_metrics['igd']:<20.4f}")
    print(f"{'Num solutions':<20} {len(random_objectives):<20} {len(nsga2_objectives):<20}")
    print("-"*70)

    # Both should find solutions
    assert random_metrics['hypervolume'] > 0
    assert nsga2_metrics['hypervolume'] > 0

    print("\n✅ Comparison tests PASSED")


def main():
    print("\n" + "#"*70)
    print("# BASELINE ALGORITHMS TEST SUITE")
    print("#"*70)

    try:
        test_random_sampler()
        test_nsga2_adapter()
        test_comparison()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()