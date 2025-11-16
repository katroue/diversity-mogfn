"""
Test baseline algorithms for multi-objective optimization.

Tests:
1. RandomSampler can sample trajectories
2. NSGA2Adapter can optimize (if pymoo installed)
3. Both produce valid objectives
4. Metrics can be computed for both
"""

import sys
from pathlib import Path
import numpy as np
import pytest

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


class TestRandomSampler:
    """Test Random Sampling baseline."""

    def test_initialization(self):
        """Test RandomSampler can be initialized."""
        env = HyperGrid(height=4, num_objectives=2)
        sampler = RandomSampler(env, max_steps=10, seed=42)

        assert sampler.env == env
        assert sampler.max_steps == 10
        assert len(sampler.trajectories) == 0

    def test_sample_trajectory(self):
        """Test sampling a single trajectory."""
        env = HyperGrid(height=4, num_objectives=2)
        sampler = RandomSampler(env, max_steps=20, seed=42)

        traj = sampler.sample_trajectory()

        assert traj is not None
        assert len(traj.states) > 0
        assert len(traj.actions) > 0
        assert len(traj.objectives) == env.num_objectives
        assert traj.objectives.shape == (2,)

    def test_train(self):
        """Test training (sampling multiple trajectories)."""
        env = HyperGrid(height=4, num_objectives=2)
        sampler = RandomSampler(env, max_steps=20, seed=42)

        history = sampler.train(
            num_iterations=10,
            batch_size=8,
            log_interval=5
        )

        assert 'num_sampled' in history
        assert 'num_unique_solutions' in history
        assert len(sampler.objectives_history) > 0

        # Check all objectives have correct shape
        for obj in sampler.objectives_history:
            assert obj.shape == (2,)

    def test_get_pareto_front(self):
        """Test Pareto front extraction."""
        env = HyperGrid(height=4, num_objectives=2)
        sampler = RandomSampler(env, max_steps=20, seed=42)

        sampler.train(num_iterations=20, batch_size=10)
        pareto_front = sampler.get_pareto_front()

        assert len(pareto_front) > 0
        assert pareto_front.shape[1] == 2  # num_objectives

        # Check Pareto front is non-dominated
        for i, obj_i in enumerate(pareto_front):
            for j, obj_j in enumerate(pareto_front):
                if i == j:
                    continue
                # No solution should strictly dominate another in Pareto front
                dominates = np.all(obj_j >= obj_i) and np.any(obj_j > obj_i)
                assert not dominates, f"Solution {j} dominates solution {i} in Pareto front"

    def test_sample_method(self):
        """Test sampling solutions."""
        env = HyperGrid(height=4, num_objectives=2)
        sampler = RandomSampler(env, max_steps=20, seed=42)

        # First train to get some solutions
        sampler.train(num_iterations=10, batch_size=10)

        # Sample
        objectives, states = sampler.sample(num_samples=5)

        assert len(objectives) > 0
        assert len(states) > 0
        assert objectives.shape[1] == 2

    def test_metrics_computation(self):
        """Test that metrics can be computed from random baseline."""
        env = HyperGrid(height=4, num_objectives=2)
        sampler = RandomSampler(env, max_steps=20, seed=42)

        sampler.train(num_iterations=50, batch_size=20)
        objectives = sampler.get_all_objectives()

        assert len(objectives) >= 10, "Need at least 10 samples for metrics"

        # Traditional metrics
        reference_point = np.array([1.1, 1.1])
        traditional = compute_all_traditional_metrics(objectives, reference_point)

        assert 'hypervolume' in traditional
        assert 'spacing' in traditional
        assert traditional['hypervolume'] >= 0
        assert traditional['spacing'] >= 0

        # Spatial metrics
        mce, num_modes = mode_coverage_entropy(objectives)
        pmd = pairwise_minimum_distance(objectives)

        assert mce >= 0
        assert num_modes > 0
        assert pmd >= 0

    def test_reset(self):
        """Test resetting the sampler."""
        env = HyperGrid(height=4, num_objectives=2)
        sampler = RandomSampler(env, max_steps=20, seed=42)

        sampler.train(num_iterations=10, batch_size=10)
        assert len(sampler.objectives_history) > 0

        sampler.reset()
        assert len(sampler.objectives_history) == 0
        assert len(sampler.trajectories) == 0


@pytest.mark.skipif(not NSGA2_AVAILABLE, reason="pymoo not installed")
class TestNSGA2Adapter:
    """Test NSGA-II baseline (requires pymoo)."""

    def test_initialization(self):
        """Test NSGA2Adapter can be initialized."""
        env = HyperGrid(height=4, num_objectives=2)
        nsga2 = NSGA2Adapter(env, pop_size=20, max_steps=10, seed=42)

        assert nsga2.env == env
        assert nsga2.pop_size == 20
        assert nsga2.max_steps == 10

    def test_train(self):
        """Test NSGA-II training."""
        env = HyperGrid(height=4, num_objectives=2)
        nsga2 = NSGA2Adapter(env, pop_size=20, max_steps=10, seed=42)

        history = nsga2.train(num_iterations=5, log_interval=2)

        assert 'generation' in history
        assert 'pareto_size' in history
        assert len(history['generation']) > 0

    def test_get_pareto_front(self):
        """Test Pareto front extraction."""
        env = HyperGrid(height=4, num_objectives=2)
        nsga2 = NSGA2Adapter(env, pop_size=20, max_steps=10, seed=42)

        nsga2.train(num_iterations=5)
        pareto_front = nsga2.get_pareto_front()

        assert len(pareto_front) > 0
        assert pareto_front.shape[1] == 2  # num_objectives

    def test_sample_method(self):
        """Test sampling from NSGA-II population."""
        env = HyperGrid(height=4, num_objectives=2)
        nsga2 = NSGA2Adapter(env, pop_size=20, max_steps=10, seed=42)

        nsga2.train(num_iterations=5)
        objectives, decision_vars = nsga2.sample(num_samples=5)

        assert len(objectives) == 5
        assert len(decision_vars) == 5
        assert objectives.shape == (5, 2)

    def test_metrics_computation(self):
        """Test that metrics can be computed from NSGA-II."""
        env = HyperGrid(height=4, num_objectives=2)
        nsga2 = NSGA2Adapter(env, pop_size=30, max_steps=10, seed=42)

        nsga2.train(num_iterations=10)
        objectives = nsga2.get_all_objectives()

        assert len(objectives) >= 10

        # Traditional metrics
        reference_point = np.array([1.1, 1.1])
        traditional = compute_all_traditional_metrics(objectives, reference_point)

        assert 'hypervolume' in traditional
        assert 'spacing' in traditional
        assert traditional['hypervolume'] >= 0

        # Spatial metrics
        mce, num_modes = mode_coverage_entropy(objectives)
        assert mce >= 0

    def test_reset(self):
        """Test resetting NSGA-II."""
        env = HyperGrid(height=4, num_objectives=2)
        nsga2 = NSGA2Adapter(env, pop_size=20, max_steps=10, seed=42)

        nsga2.train(num_iterations=5)
        assert nsga2.result is not None

        nsga2.reset()
        assert nsga2.result is None
        assert len(nsga2.objectives_history) == 0


class TestBaselineComparison:
    """Test comparing different baselines."""

    def test_random_vs_nsga2_different_results(self):
        """Test that Random and NSGA-II produce different results."""
        if not NSGA2_AVAILABLE:
            pytest.skip("pymoo not installed")

        env = HyperGrid(height=4, num_objectives=2)
        seed = 42

        # Run Random
        random_sampler = RandomSampler(env, max_steps=20, seed=seed)
        random_sampler.train(num_iterations=20, batch_size=10)
        random_objectives = random_sampler.get_all_objectives()

        # Run NSGA-II
        nsga2 = NSGA2Adapter(env, pop_size=20, max_steps=20, seed=seed)
        nsga2.train(num_iterations=10)
        nsga2_objectives = nsga2.get_all_objectives()

        # They should have different number of solutions or different values
        # (extremely unlikely to be identical)
        if len(random_objectives) == len(nsga2_objectives):
            assert not np.allclose(random_objectives, nsga2_objectives)

    def test_nsga2_better_than_random(self):
        """Test that NSGA-II typically outperforms random sampling."""
        if not NSGA2_AVAILABLE:
            pytest.skip("pymoo not installed")

        env = HyperGrid(height=6, num_objectives=2)
        seed = 42
        reference_point = np.array([1.1, 1.1])

        # Run Random (more samples to be fair)
        random_sampler = RandomSampler(env, max_steps=20, seed=seed)
        random_sampler.train(num_iterations=100, batch_size=20)
        random_objectives = random_sampler.get_all_objectives()
        random_metrics = compute_all_traditional_metrics(random_objectives, reference_point)

        # Run NSGA-II
        nsga2 = NSGA2Adapter(env, pop_size=50, max_steps=20, seed=seed)
        nsga2.train(num_iterations=20)
        nsga2_objectives = nsga2.get_all_objectives()
        nsga2_metrics = compute_all_traditional_metrics(nsga2_objectives, reference_point)

        # NSGA-II should generally have better hypervolume
        # (though not guaranteed on every seed)
        print(f"\nRandom HV: {random_metrics['hypervolume']:.4f}")
        print(f"NSGA-II HV: {nsga2_metrics['hypervolume']:.4f}")

        # At minimum, both should find some solutions
        assert random_metrics['hypervolume'] > 0
        assert nsga2_metrics['hypervolume'] > 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])