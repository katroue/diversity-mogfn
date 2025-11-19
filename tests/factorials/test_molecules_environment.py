#!/usr/bin/env python3
"""
Test script for Molecule environment.

Tests the MoleculeFragments environment implementation and its integration with MOGFN.

Usage:
    python tests/test_molecules_environment.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.molecules import MoleculeFragments
from src.models.mogfn_pc import MOGFN_PC, PreferenceSampler, MOGFNTrainer, MOGFNSampler


def test_environment_interface():
    """Test that MoleculeFragments implements the required interface."""
    print("\n" + "="*80)
    print("TEST 1: Environment Interface")
    print("="*80)

    env = MoleculeFragments(max_fragments=6, num_fragments_library=10)

    # Check required properties
    assert hasattr(env, 'state_dim'), "Missing state_dim property"
    assert hasattr(env, 'num_actions'), "Missing num_actions property"
    assert hasattr(env, 'num_objectives'), "Missing num_objectives property"

    print(f"✓ state_dim: {env.state_dim}")
    print(f"✓ num_actions: {env.num_actions}")
    print(f"✓ num_objectives: {env.num_objectives}")

    # Check required methods
    assert hasattr(env, 'get_initial_state'), "Missing get_initial_state method"
    assert hasattr(env, 'step'), "Missing step method"
    assert hasattr(env, 'get_valid_actions'), "Missing get_valid_actions method"
    assert hasattr(env, 'compute_objectives'), "Missing compute_objectives method"

    print("✓ All required methods present")

    print("\n✓ TEST 1 PASSED: Environment interface complete\n")


def test_molecule_construction():
    """Test that molecules are constructed correctly."""
    print("\n" + "="*80)
    print("TEST 2: Molecule Construction")
    print("="*80)

    env = MoleculeFragments(max_fragments=5, num_fragments_library=10)

    # Build a specific molecule
    state = env.get_initial_state()
    target_fragments = [3, 6, 7]  # Benzene, Amine, Hydroxyl

    for i, frag_idx in enumerate(target_fragments):
        # Check valid actions
        valid_actions = env.get_valid_actions(state)
        assert frag_idx in valid_actions, f"Fragment {frag_idx} not valid at step {i}"

        # Take action
        state, done = env.step(state, frag_idx)
        current_smiles = env._state_to_smiles(state)

        print(f"  Step {i+1}: Added {env.get_fragment_name(frag_idx)}, SMILES='{current_smiles}', done={done}")

    # Verify final molecule
    final_smiles = env._state_to_smiles(state)
    num_frags = int(state[0].item())
    assert num_frags == len(target_fragments), f"Expected {len(target_fragments)} fragments, got {num_frags}"

    print(f"\n✓ Built molecule: '{final_smiles}'")
    print(f"✓ Fragment count: {num_frags}")
    print("✓ TEST 2 PASSED: Molecule construction works correctly\n")


def test_objective_computation():
    """Test that objectives are computed correctly."""
    print("\n" + "="*80)
    print("TEST 3: Objective Computation")
    print("="*80)

    # Test with simple heuristic objectives (no RDKit needed)
    env = MoleculeFragments(
        max_fragments=6,
        num_fragments_library=10,
        objective_properties=['length', 'diversity']
    )

    # Test molecule with all same fragments
    state = env.get_initial_state()
    fragments = [0, 0, 0, 0]  # Four methyls

    for frag in fragments:
        state, _ = env.step(state, frag)

    objectives = env.compute_objectives(state)

    print(f"Molecule: Four methyls")
    print(f"SMILES: '{env._state_to_smiles(state)}'")
    print(f"Objectives: {objectives}")

    length_score = objectives[0].item()
    diversity_score = objectives[1].item()

    # Length should be 4/6
    expected_length = 4.0 / env.max_fragments
    assert abs(length_score - expected_length) < 0.01, \
        f"Expected length {expected_length}, got {length_score}"

    # Diversity should be low (all same fragment) = 1/4 = 0.25
    expected_diversity = 0.25
    assert abs(diversity_score - expected_diversity) < 0.01, \
        f"Expected diversity {expected_diversity}, got {diversity_score}"

    print(f"  ✓ Length: {length_score:.3f} (expected {expected_length:.3f})")
    print(f"  ✓ Diversity: {diversity_score:.3f} (expected {expected_diversity:.3f})")

    # Test molecule with all different fragments
    state = env.get_initial_state()
    fragments = [0, 1, 2, 3]  # All different

    for frag in fragments:
        state, _ = env.step(state, frag)

    objectives = env.compute_objectives(state)
    diversity_score = objectives[1].item()

    # Diversity should be 1.0 (all different)
    assert diversity_score == 1.0, f"Expected diversity 1.0, got {diversity_score}"
    print(f"  ✓ Diversity (all different): {diversity_score:.3f}")

    print("\n✓ TEST 3 PASSED: Objective computation is correct\n")


def test_mogfn_integration():
    """Test integration with MOGFN model."""
    print("\n" + "="*80)
    print("TEST 4: MOGFN Integration")
    print("="*80)

    # Create environment
    env = MoleculeFragments(
        max_fragments=5,
        num_fragments_library=8,
        objective_properties=['length', 'diversity']
    )

    # Create MOGFN model
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=32,
        num_actions=env.num_actions,
        num_layers=2,
        preference_encoding='vanilla',
        conditioning_type='concat',
        temperature=2.0
    )

    print(f"✓ Created MOGFN with {sum(p.numel() for p in mogfn.parameters())} parameters")

    # Create preference sampler
    pref_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution='dirichlet',
        alpha=1.5
    )

    # Create sampler
    sampler = MOGFNSampler(mogfn, env, pref_sampler)

    # Sample a trajectory
    preference = pref_sampler.sample(1)[0]
    print(f"✓ Sampled preference: {preference}")

    trajectory = sampler.sample_trajectory(preference, explore=True)

    print(f"✓ Sampled trajectory with {len(trajectory.states)} states")
    print(f"✓ Actions: {trajectory.actions}")
    print(f"✓ Terminal: {trajectory.is_terminal}")

    # Extract final molecule
    final_state = trajectory.states[-1]
    smiles = env._state_to_smiles(final_state)
    print(f"✓ Generated molecule: '{smiles}'")

    # Check objectives
    objectives = trajectory.reward
    print(f"✓ Objectives: {objectives}")

    print("\n✓ TEST 4 PASSED: MOGFN integration works\n")


def test_training():
    """Test training MOGFN on Molecule environment."""
    print("\n" + "="*80)
    print("TEST 5: Training (50 iterations)")
    print("="*80)

    # Create environment
    env = MoleculeFragments(
        max_fragments=5,
        num_fragments_library=8,
        objective_properties=['length', 'diversity']
    )

    # Create MOGFN model
    mogfn = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=32,
        num_actions=env.num_actions,
        num_layers=2,
        preference_encoding='vanilla',
        conditioning_type='concat',
        temperature=2.0
    )

    # Create preference sampler
    pref_sampler = PreferenceSampler(
        num_objectives=env.num_objectives,
        distribution='dirichlet',
        alpha=1.5
    )

    # Create optimizer
    optimizer = torch.optim.Adam(mogfn.parameters(), lr=0.001)

    # Create trainer
    trainer = MOGFNTrainer(
        mogfn=mogfn,
        env=env,
        preference_sampler=pref_sampler,
        optimizer=optimizer,
        loss_function='trajectory_balance',
        loss_params={},
        regularization='none',
        regularization_params={},
        gradient_clip=10.0
    )

    print("✓ Created trainer")

    # Train for a few iterations
    num_iterations = 50
    history = trainer.train(
        num_iterations=num_iterations,
        batch_size=16,
        num_preferences_per_batch=4,
        log_every=25
    )

    print(f"✓ Trained for {num_iterations} iterations")
    print(f"  Initial loss: {history['loss'][0]:.4f}")
    print(f"  Final loss: {history['loss'][-1]:.4f}")

    # Evaluate
    eval_results = trainer.evaluate(num_samples=20)

    objectives = eval_results['objectives']
    print(f"✓ Evaluated {len(objectives)} samples")
    print(f"  Objective statistics:")

    objectives_np = objectives.detach().cpu().numpy()
    for i in range(env.num_objectives):
        mean = objectives_np[:, i].mean()
        std = objectives_np[:, i].std()
        print(f"    {env.objective_properties[i]}: {mean:.3f} ± {std:.3f}")

    print("\n✓ TEST 5 PASSED: Training works\n")


def test_rdkit_properties():
    """Test molecular property computation with RDKit (if available)."""
    print("\n" + "="*80)
    print("TEST 6: RDKit Properties (Optional)")
    print("="*80)

    try:
        import rdkit
        print("✓ RDKit is available")

        # Create environment with molecular properties
        env = MoleculeFragments(
            max_fragments=5,
            num_fragments_library=10,
            objective_properties=['qed', 'sa', 'logp'],
            use_rdkit=True
        )

        if not env._rdkit_available:
            print("⚠ RDKit features not fully available")
            print("✓ TEST 6 SKIPPED\n")
            return

        # Build a test molecule
        state = env.get_initial_state()
        fragments = [3, 6, 7]  # Benzene-Amine-Hydroxyl

        for frag in fragments:
            state, _ = env.step(state, frag)

        smiles = env._state_to_smiles(state)
        objectives = env.compute_objectives(state)

        print(f"Molecule: '{smiles}'")
        print(f"Properties:")
        for i, prop in enumerate(env.objective_properties):
            print(f"  {prop.upper()}: {objectives[i].item():.3f}")

        # Verify all properties are in valid range [0, 1]
        for i, prop in enumerate(env.objective_properties):
            value = objectives[i].item()
            assert 0.0 <= value <= 1.0, f"{prop} value {value} out of range [0,1]"
            print(f"  ✓ {prop} in valid range")

        print("\n✓ TEST 6 PASSED: RDKit properties work\n")

    except ImportError:
        print("⚠ RDKit not available - skipping RDKit-specific tests")
        print("  Install RDKit with: conda install -c conda-forge rdkit")
        print("✓ TEST 6 SKIPPED\n")


def test_fragment_library():
    """Test the fragment library and SMILES generation."""
    print("\n" + "="*80)
    print("TEST 7: Fragment Library")
    print("="*80)

    env = MoleculeFragments(max_fragments=4, num_fragments_library=15)

    print(f"Fragment library size: {env.num_fragments_library}")
    print(f"\nFragment library:")

    for i in range(min(10, env.num_fragments_library)):
        smiles = env.FRAGMENT_LIBRARY[i]
        name = env.get_fragment_name(i)
        print(f"  {i:2d}. {name:20s} - {smiles}")

    # Test building molecules with different fragments
    test_cases = [
        ([0, 1, 2], "Alkyl chain"),
        ([3, 6], "Aromatic amine"),
        ([3, 4, 5], "Aromatic with carbonyl/carboxyl"),
    ]

    print(f"\nTest molecules:")
    for fragments, description in test_cases:
        state = env.get_initial_state()
        for frag in fragments:
            state, _ = env.step(state, frag)

        smiles = env._state_to_smiles(state)
        fragment_names = [env.get_fragment_name(f) for f in fragments]

        print(f"  {description}:")
        print(f"    Fragments: {fragment_names}")
        print(f"    SMILES: '{smiles}'")

    print("\n✓ TEST 7 PASSED: Fragment library works\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MOLECULE ENVIRONMENT TEST SUITE")
    print("="*80)

    try:
        test_environment_interface()
        test_molecule_construction()
        test_objective_computation()
        test_fragment_library()
        test_mogfn_integration()
        test_training()
        test_rdkit_properties()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nThe Molecule environment is working correctly:")
        print("  • Implements required interface")
        print("  • Constructs molecules properly")
        print("  • Computes objectives correctly")
        print("  • Fragment library functional")
        print("  • Integrates with MOGFN model")
        print("  • Supports training")
        print("  • RDKit integration available (if installed)")
        print("\n")

    except AssertionError as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}\n")
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST ERROR")
        print("="*80)
        print(f"\nUnexpected error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
