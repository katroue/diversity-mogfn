#!/usr/bin/env python3
"""
Test script for DNA Sequence environment.

Tests the DNASequence environment implementation and its integration with MOGFN.

Usage:
    python tests/test_dna_environment.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.sequences import DNASequence
from src.models.mogfn_pc import MOGFN_PC, PreferenceSampler, MOGFNTrainer, MOGFNSampler


def test_environment_interface():
    """Test that DNASequence implements the required interface."""
    print("\n" + "="*80)
    print("TEST 1: Environment Interface")
    print("="*80)

    env = DNASequence(seq_length=15, objectives=['free_energy', 'num_base_pairs'])

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


def test_sequence_construction():
    """Test that sequences are constructed correctly."""
    print("\n" + "="*80)
    print("TEST 2: Sequence Construction")
    print("="*80)

    env = DNASequence(seq_length=10)

    # Build a specific sequence
    state = env.get_initial_state()
    target_sequence = "ACGTTATA"

    print(f"Building sequence: {target_sequence}")

    for i, base in enumerate(target_sequence):
        # Check valid actions
        valid_actions = env.get_valid_actions(state)
        action = env.BASE_TO_IDX[base]
        assert action in valid_actions, f"Base {base} (action {action}) not valid at step {i}"

        # Take action
        state, done = env.step(state, action)
        current_sequence = env._state_to_sequence(state)

        print(f"  Step {i+1}: Added {base}, sequence='{current_sequence}', done={done}")

    # Verify final sequence
    final_sequence = env._state_to_sequence(state)
    length = env.get_sequence_length(state)

    assert final_sequence == target_sequence, f"Expected '{target_sequence}', got '{final_sequence}'"
    assert length == len(target_sequence), f"Expected length {len(target_sequence)}, got {length}"

    print(f"\n✓ Built sequence: '{final_sequence}'")
    print(f"✓ Length: {length}")
    print("✓ TEST 2 PASSED: Sequence construction works correctly\n")


def test_objective_computation():
    """Test that objectives are computed correctly (paper specification)."""
    print("\n" + "="*80)
    print("TEST 3: Objective Computation (Paper Objectives)")
    print("="*80)

    env = DNASequence(
        seq_length=20,
        objectives=['free_energy', 'num_base_pairs', 'inverse_length'],
        use_viennarna=False  # Use heuristics for testing
    )

    # Test 1: Structured sequence with palindrome (can form hairpin)
    print("\nTest 1: Palindromic sequence (GCATATGC)")
    print("  Should form structure with base pairs")
    state = env.get_initial_state()
    for base in "GCATATGC":
        state, _ = env.step(state, env.BASE_TO_IDX[base])

    objectives = env.compute_objectives(state)
    free_energy = objectives[0].item()
    num_pairs = objectives[1].item()
    inv_length = objectives[2].item()

    print(f"  Free energy score: {free_energy:.4f}")
    print(f"  Num base pairs score: {num_pairs:.4f} (should have potential pairs)")
    print(f"  Inverse length: {inv_length:.4f} (length=8, so 1/8=0.125)")

    # Should have some base pairing potential
    assert num_pairs > 0.0, f"Expected some base pairs, got {num_pairs}"
    assert abs(inv_length - 0.125) < 0.01, f"Expected inv_length=0.125, got {inv_length}"

    # Test 2: All complementary bases (ATATATAT)
    print("\nTest 2: Complementary sequence (ATATATAT)")
    print("  All AT pairs, many potential pairs")
    state = env.get_initial_state()
    for base in "ATATATAT":
        state, _ = env.step(state, env.BASE_TO_IDX[base])

    objectives = env.compute_objectives(state)
    free_energy = objectives[0].item()
    num_pairs = objectives[1].item()
    inv_length = objectives[2].item()

    print(f"  Free energy score: {free_energy:.4f}")
    print(f"  Num base pairs score: {num_pairs:.4f} (4A + 4T → 4 potential pairs)")
    print(f"  Inverse length: {inv_length:.4f}")

    assert num_pairs == 1.0, f"Expected max base pairs (1.0), got {num_pairs}"

    # Test 3: Inverse length objective
    print("\nTest 3: Inverse length objective (different lengths)")
    for length in [2, 5, 10]:
        state = env.get_initial_state()
        for _ in range(length):
            state, _ = env.step(state, 0)  # Add 'A'

        objectives = env.compute_objectives(state)
        inv_length = objectives[2].item()
        expected = 1.0 / length

        print(f"  Length {length}: inverse = {inv_length:.4f} (expected {expected:.4f})")
        assert abs(inv_length - expected) < 0.01, \
            f"Expected {expected}, got {inv_length}"

    print("\n✓ TEST 3 PASSED: Objective computation is correct\n")


def test_objective_tradeoffs():
    """Test objective trade-offs (paper specification)."""
    print("\n" + "="*80)
    print("TEST 4: Objective Trade-offs")
    print("="*80)

    env = DNASequence(
        seq_length=15,
        objectives=['free_energy', 'num_base_pairs', 'inverse_length'],
        use_viennarna=False
    )

    # Test trade-off: longer sequence = more base pairs but lower inverse_length
    print("\nTest 1: Length vs. base pairs trade-off")

    for length in [3, 6, 12]:
        state = env.get_initial_state()
        # Create sequence with complementary bases
        for i in range(length):
            base = 'A' if i % 2 == 0 else 'T'
            state, _ = env.step(state, env.BASE_TO_IDX[base])

        objectives = env.compute_objectives(state)
        num_pairs = objectives[1].item()
        inv_length = objectives[2].item()

        print(f"  Length {length}: pairs={num_pairs:.3f}, inv_length={inv_length:.3f}")

        # Longer sequences should have more base pairs but lower inverse length
        expected_inv = 1.0 / length
        assert abs(inv_length - expected_inv) < 0.01, f"Inverse length incorrect: expected {expected_inv}, got {inv_length}"

    # Test: GC-rich vs AT-rich (affects free energy heuristic)
    print("\nTest 2: GC vs. AT content (affects free energy)")

    # GC-rich sequence
    state_gc = env.get_initial_state()
    for base in "GCGCGCGC":
        state_gc, _ = env.step(state_gc, env.BASE_TO_IDX[base])

    # AT-rich sequence
    state_at = env.get_initial_state()
    for base in "ATATATAT":
        state_at, _ = env.step(state_at, env.BASE_TO_IDX[base])

    obj_gc = env.compute_objectives(state_gc)
    obj_at = env.compute_objectives(state_at)

    print(f"  GC-rich: free_energy={obj_gc[0]:.4f}, pairs={obj_gc[1]:.4f}")
    print(f"  AT-rich: free_energy={obj_at[0]:.4f}, pairs={obj_at[1]:.4f}")

    # Both should have perfect pairing, but GC is more stable
    assert obj_gc[1] == obj_at[1] == 1.0, "Both should have perfect pairing"
    # GC-rich should have higher free energy score (more stable)
    assert obj_gc[0] >= obj_at[0], "GC-rich should be more stable"

    print("\n✓ TEST 4 PASSED: Objective trade-offs work correctly\n")


def test_mogfn_integration():
    """Test integration with MOGFN model."""
    print("\n" + "="*80)
    print("TEST 5: MOGFN Integration")
    print("="*80)

    # Create environment
    env = DNASequence(
        seq_length=12,
        objectives=['free_energy', 'inverse_length'],
        use_viennarna=False
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

    # Extract final sequence
    final_state = trajectory.states[-1]
    sequence = env._state_to_sequence(final_state)
    print(f"✓ Generated sequence: '{sequence}'")

    # Check objectives
    objectives = trajectory.reward
    print(f"✓ Objectives: {objectives}")

    print("\n✓ TEST 5 PASSED: MOGFN integration works\n")


def test_training():
    """Test training MOGFN on DNA environment."""
    print("\n" + "="*80)
    print("TEST 6: Training (50 iterations)")
    print("="*80)

    # Create environment
    env = DNASequence(
        seq_length=10,
        objectives=['num_base_pairs', 'inverse_length'],
        use_viennarna=False
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
        print(f"    {env.objectives[i]}: {mean:.3f} ± {std:.3f}")

    print("\n✓ TEST 6 PASSED: Training works\n")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*80)
    print("TEST 7: Edge Cases")
    print("="*80)

    env = DNASequence(seq_length=5, objectives=['num_base_pairs', 'inverse_length'], use_viennarna=False)

    # Test 1: Empty sequence
    print("\nTest 1: Empty sequence")
    state = env.get_initial_state()
    objectives = env.compute_objectives(state)
    print(f"  Empty sequence objectives: {objectives}")
    assert torch.all(objectives == 0.0), "Empty sequence should have zero objectives"
    print("  ✓ Empty sequence handled correctly")

    # Test 2: Single base
    print("\nTest 2: Single base sequence")
    state = env.get_initial_state()
    state, _ = env.step(state, 0)  # Add 'A'
    sequence = env._state_to_sequence(state)
    objectives = env.compute_objectives(state)
    print(f"  Sequence: '{sequence}'")
    print(f"  Objectives: {objectives}")
    assert len(sequence) == 1, "Should have exactly one base"
    print("  ✓ Single base handled correctly")

    # Test 3: Maximum length sequence
    print("\nTest 3: Maximum length sequence")
    state = env.get_initial_state()
    for _ in range(env.seq_length):
        valid_actions = env.get_valid_actions(state)
        # Choose a base action (not DONE)
        action = [a for a in valid_actions if a != env.DONE_ACTION][0]
        state, done = env.step(state, action)

    sequence = env._state_to_sequence(state)
    print(f"  Sequence: '{sequence}'")
    print(f"  Length: {len(sequence)} / {env.seq_length}")
    assert len(sequence) == env.seq_length, "Should be at maximum length"
    print("  ✓ Maximum length handled correctly")

    # Test 4: Early termination
    print("\nTest 4: Early termination (DONE action)")
    state = env.get_initial_state()
    state, _ = env.step(state, 0)  # Add 'A'
    state, _ = env.step(state, 2)  # Add 'G'
    state, done = env.step(state, env.DONE_ACTION)  # Terminate
    sequence = env._state_to_sequence(state)
    print(f"  Sequence: '{sequence}'")
    print(f"  Done: {done}")
    assert done == True, "Should be terminated"
    assert len(sequence) == 2, "Should have only 2 bases"
    print("  ✓ Early termination works correctly")

    print("\n✓ TEST 7 PASSED: Edge cases handled correctly\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DNA SEQUENCE ENVIRONMENT TEST SUITE")
    print("="*80)

    try:
        test_environment_interface()
        test_sequence_construction()
        test_objective_computation()
        test_objective_tradeoffs()
        test_mogfn_integration()
        test_training()
        test_edge_cases()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nThe DNA Sequence environment is working correctly:")
        print("  • Implements required interface")
        print("  • Constructs sequences properly")
        print("  • Computes objectives correctly (paper spec: free energy, base pairs, length)")
        print("  • Tests objective trade-offs")
        print("  • Integrates with MOGFN model")
        print("  • Supports training")
        print("  • Handles edge cases")
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
