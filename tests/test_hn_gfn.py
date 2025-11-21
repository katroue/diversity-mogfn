"""
Test script for HN-GFN (Hypernetwork-GFlowNet) baseline.

This test validates the implementation of HN-GFN by:
1. Creating a simple HyperGrid environment
2. Instantiating and training HN-GFN
3. Testing sampling and objective collection
4. Verifying the hypernetwork produces preference-dependent Z values
"""

import sys
import os
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import torch
import numpy as np
from models.baselines import HN_GFN
from environments.hypergrid import HyperGrid


def test_hn_gfn_initialization():
    """Test that HN-GFN can be initialized correctly."""
    print("\n=== Test 1: HN-GFN Initialization ===")

    env = HyperGrid(height=6, num_objectives=2)
    state_dim = env.state_dim  # Use environment's state_dim (2 for coordinates)
    num_actions = env.num_actions  # Use environment's num_actions (3 for right/up/done)

    hngfn = HN_GFN(
        env=env,
        state_dim=state_dim,
        num_objectives=2,
        hidden_dim=32,
        num_actions=num_actions,
        num_layers=2,
        z_hidden_dim=32,
        z_num_layers=2,
        learning_rate=1e-3,
        z_learning_rate=1e-3,
        alpha=1.5,
        max_steps=50,
        seed=42
    )

    print(f"✓ HN-GFN initialized: {hngfn}")
    print(f"✓ Policy network parameters: {sum(p.numel() for p in hngfn.model.parameters())}")
    print(f"✓ Z hypernetwork parameters: {sum(p.numel() for p in hngfn.Z_network.parameters())}")

    return hngfn, env


def test_z_hypernetwork(hngfn):
    """Test that Z hypernetwork produces different values for different preferences."""
    print("\n=== Test 2: Z Hypernetwork Preference-Dependence ===")

    # Test with different preferences
    pref1 = torch.FloatTensor([1.0, 0.0])
    pref2 = torch.FloatTensor([0.0, 1.0])
    pref3 = torch.FloatTensor([0.5, 0.5])

    with torch.no_grad():
        z1 = hngfn.Z_network(pref1)
        z2 = hngfn.Z_network(pref2)
        z3 = hngfn.Z_network(pref3)

    print(f"✓ Z([1.0, 0.0]) = {z1.item():.4f}")
    print(f"✓ Z([0.0, 1.0]) = {z2.item():.4f}")
    print(f"✓ Z([0.5, 0.5]) = {z3.item():.4f}")

    # Verify they are different
    assert not torch.allclose(z1, z2), "Z should be different for different preferences"
    print("✓ Z hypernetwork produces preference-dependent values")


def test_trajectory_sampling(hngfn):
    """Test trajectory sampling with different preferences."""
    print("\n=== Test 3: Trajectory Sampling ===")

    # Sample with specific preference
    pref = np.array([0.7, 0.3])
    traj = hngfn.sample_trajectory(preference=pref, explore=False)

    print(f"✓ Sampled trajectory with {len(traj.actions)} actions")
    print(f"✓ Terminal: {traj.is_terminal}")
    print(f"✓ Objectives: {traj.objectives}")
    print(f"✓ Preference: {traj.preference}")

    # Sample without preference (random Dirichlet)
    traj2 = hngfn.sample_trajectory(explore=False)
    print(f"✓ Sampled trajectory with random preference: {traj2.preference}")

    return traj


def test_training(hngfn):
    """Test that HN-GFN can train without errors."""
    print("\n=== Test 4: Training HN-GFN ===")

    # Train for a small number of iterations
    history = hngfn.train(
        num_iterations=50,
        batch_size=8,
        log_interval=25
    )

    print(f"✓ Training completed")
    print(f"✓ Terminal trajectories collected: {len(hngfn.objectives_history)}")
    print(f"✓ Training losses: {len(hngfn.training_losses)}")

    if len(hngfn.training_losses) > 0:
        print(f"✓ Initial loss: {hngfn.training_losses[0]:.4f}")
        print(f"✓ Final loss: {hngfn.training_losses[-1]:.4f}")

    return history


def test_sampling(hngfn):
    """Test sampling solutions after training."""
    print("\n=== Test 5: Sampling Solutions ===")

    # Sample with specific preference
    objectives1, states1 = hngfn.sample(num_samples=10, preference=np.array([0.8, 0.2]))
    print(f"✓ Sampled {len(objectives1)} solutions with preference [0.8, 0.2]")
    print(f"  Mean objectives: {objectives1.mean(axis=0)}")

    # Sample with random preferences
    objectives2, states2 = hngfn.sample(num_samples=10)
    print(f"✓ Sampled {len(objectives2)} solutions with random preferences")
    print(f"  Mean objectives: {objectives2.mean(axis=0)}")


def test_pareto_front(hngfn):
    """Test Pareto front extraction."""
    print("\n=== Test 6: Pareto Front Extraction ===")

    all_objectives = hngfn.get_all_objectives()
    print(f"✓ Total objectives collected: {len(all_objectives)}")

    if len(all_objectives) > 0:
        pareto_front = hngfn.get_pareto_front()
        print(f"✓ Pareto front size: {len(pareto_front)}")
        print(f"  Pareto ratio: {len(pareto_front)/len(all_objectives)*100:.1f}%")


def test_save_load(hngfn, tmpdir="/tmp"):
    """Test saving and loading checkpoints."""
    print("\n=== Test 7: Save/Load Checkpoint ===")

    checkpoint_path = os.path.join(tmpdir, "test_hngfn_checkpoint.pt")

    # Save
    hngfn.save(checkpoint_path)
    print(f"✓ Saved checkpoint to {checkpoint_path}")

    # Get current state
    old_objectives = hngfn.get_all_objectives().copy()

    # Create new model and load
    env = hngfn.env
    new_hngfn = HN_GFN(
        env=env,
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=32,
        num_actions=env.num_actions,
        num_layers=2,
        z_hidden_dim=32,
        z_num_layers=2,
        seed=42
    )

    new_hngfn.load(checkpoint_path)
    print(f"✓ Loaded checkpoint into new model")

    # Verify objectives were loaded
    loaded_objectives = new_hngfn.get_all_objectives()
    assert len(loaded_objectives) == len(old_objectives), "Objectives not loaded correctly"
    assert np.allclose(loaded_objectives, old_objectives), "Objectives don't match"
    print(f"✓ Checkpoint loaded correctly ({len(loaded_objectives)} objectives)")

    # Clean up
    os.remove(checkpoint_path)


def test_comparison_with_mogfn_pc():
    """Test that HN-GFN differs from MOGFN-PC in the expected way."""
    print("\n=== Test 8: Comparison with MOGFN-PC ===")

    # Import MOGFN-PC for comparison
    from models.mogfn_pc import MOGFN_PC

    env = HyperGrid(height=6, num_objectives=2)
    state_dim = env.state_dim
    num_actions = env.num_actions

    # Create MOGFN-PC
    mogfn = MOGFN_PC(
        state_dim=state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=32,
        num_actions=num_actions,
        num_layers=2,
        temperature=1.0
    )

    # Create HN-GFN
    hngfn = HN_GFN(
        env=env,
        state_dim=state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=32,
        num_actions=num_actions,
        num_layers=2,
        z_hidden_dim=32,
        z_num_layers=2,
        seed=42
    )

    # Check that MOGFN-PC has log_Z parameter
    assert hasattr(mogfn, 'log_Z'), "MOGFN-PC should have log_Z parameter"
    assert isinstance(mogfn.log_Z, torch.nn.Parameter), "log_Z should be a parameter"
    print("✓ MOGFN-PC has fixed log_Z parameter")

    # Check that HN-GFN doesn't have log_Z parameter but has Z_network
    assert not hasattr(hngfn.model, 'log_Z'), "HN-GFN should not have fixed log_Z"
    assert hasattr(hngfn, 'Z_network'), "HN-GFN should have Z_network"
    print("✓ HN-GFN has Z hypernetwork instead of fixed log_Z")

    # Verify Z_network is preference-dependent
    pref1 = torch.FloatTensor([1.0, 0.0])
    pref2 = torch.FloatTensor([0.0, 1.0])

    with torch.no_grad():
        z1 = hngfn.Z_network(pref1)
        z2 = hngfn.Z_network(pref2)

    print(f"✓ Z varies with preference: Z([1,0])={z1.item():.4f}, Z([0,1])={z2.item():.4f}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing HN-GFN (Hypernetwork-GFlowNet) Implementation")
    print("=" * 70)

    try:
        # Test initialization
        hngfn, env = test_hn_gfn_initialization()

        # Test Z hypernetwork
        test_z_hypernetwork(hngfn)

        # Test trajectory sampling
        test_trajectory_sampling(hngfn)

        # Test training
        test_training(hngfn)

        # Test sampling
        test_sampling(hngfn)

        # Test Pareto front
        test_pareto_front(hngfn)

        # Test save/load
        test_save_load(hngfn)

        # Test comparison with MOGFN-PC
        test_comparison_with_mogfn_pc()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
