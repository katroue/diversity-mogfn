"""
Inspect trained policy logits to verify mode collapse hypothesis.

This script loads a trained MOGFN-PC model and examines the actual logit
distributions during sampling to confirm why nucleus sampling fails.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from models.mogfn_pc import MOGFN_PC
from environments.hypergrid import HyperGrid


def inspect_policy_logits(checkpoint_path, num_samples=20):
    """Load model and inspect logits during trajectory sampling"""

    print("=" * 80)
    print(f"INSPECTING TRAINED POLICY: {checkpoint_path}")
    print("=" * 80)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create environment
    env = HyperGrid(height=8, num_objectives=2, reward_config='corners')

    # Create model with same config
    model = MOGFN_PC(
        state_dim=env.state_dim,
        num_objectives=env.num_objectives,
        hidden_dim=checkpoint.get('hidden_dim', 64),
        num_actions=env.num_actions,
        num_layers=checkpoint.get('num_layers', 3),
        conditioning_type=checkpoint.get('conditioning', 'film'),
        sampling_strategy=checkpoint.get('sampling_strategy', 'categorical'),
        temperature=checkpoint.get('temperature', 1.0),
        top_p=checkpoint.get('top_p', None)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nModel config:")
    print(f"  Sampling strategy: {model.sampling_strategy}")
    print(f"  Temperature: {model.temperature}")
    print(f"  Top-p: {model.top_p}")
    print(f"  Hidden dim: {checkpoint.get('hidden_dim')}")
    print(f"  Num layers: {checkpoint.get('num_layers')}")

    # Sample preferences
    preferences = torch.from_numpy(env.sample_preferences(num_samples)).float()

    print(f"\n\nSampling {num_samples} trajectories and inspecting logits...")
    print("=" * 80)

    all_logits = []
    all_states = []
    final_states = []

    for i in range(num_samples):
        preference = preferences[i]
        state = env.reset()
        done = False
        step = 0

        trajectory_logits = []

        while not done:
            state_tensor = torch.from_numpy(state).float()

            # Get logits for this state
            with torch.no_grad():
                logits = model.forward_logits(state_tensor, preference)

                # Mask invalid actions
                valid_actions = env.get_valid_actions(state)
                if valid_actions is not None:
                    mask = torch.full_like(logits, float('-inf'))
                    mask[valid_actions] = 0
                    logits = logits + mask

            trajectory_logits.append(logits.numpy())
            all_logits.append(logits.numpy())
            all_states.append(state.copy())

            # Sample action
            action, _ = model.sample_action(state_tensor, preference, valid_actions)

            # Step environment
            next_state, reward, done, info = env.step(state, action)
            state = next_state
            step += 1

            if done:
                final_states.append(state.copy())

    # Analyze logits
    all_logits = np.array(all_logits)

    print(f"\nTotal steps collected: {len(all_logits)}")
    print(f"\nLogit statistics (across all steps and trajectories):")
    print(f"  Mean max logit: {np.max(all_logits, axis=1).mean():.2f}")
    print(f"  Std max logit:  {np.max(all_logits, axis=1).std():.2f}")
    print(f"  Mean min logit: {np.min(all_logits, axis=1).mean():.2f}")
    print(f"  Mean logit range: {(np.max(all_logits, axis=1) - np.min(all_logits, axis=1)).mean():.2f}")

    # Compute softmax probabilities
    probs = []
    for logits in all_logits:
        # Remove -inf values
        finite_mask = np.isfinite(logits)
        if finite_mask.any():
            finite_logits = logits[finite_mask]
            p = np.exp(finite_logits) / np.exp(finite_logits).sum()
            max_p = p.max()
            probs.append(max_p)

    probs = np.array(probs)

    print(f"\nMax probability statistics (after softmax):")
    print(f"  Mean max prob: {probs.mean():.4f}")
    print(f"  Median max prob: {np.median(probs):.4f}")
    print(f"  % steps with max_prob > 0.9: {(probs > 0.9).mean() * 100:.1f}%")
    print(f"  % steps with max_prob > 0.99: {(probs > 0.99).mean() * 100:.1f}%")
    print(f"  % steps with max_prob > 0.999: {(probs > 0.999).mean() * 100:.1f}%")

    # Analyze final states
    final_states = np.array(final_states)
    unique_finals = np.unique(final_states, axis=0)

    print(f"\n\nFinal state diversity:")
    print(f"  Total trajectories: {len(final_states)}")
    print(f"  Unique final states: {len(unique_finals)}")
    print(f"  Mode coverage: {len(unique_finals) / len(final_states) * 100:.1f}%")

    if len(unique_finals) <= 5:
        print(f"\n  Unique final states:")
        for i, state in enumerate(unique_finals):
            count = (final_states == state).all(axis=1).sum()
            objectives = env.get_objectives(state)
            print(f"    State {i+1}: {state} -> objectives {objectives}, count={count}")

    # Show example of very peaked distribution
    print(f"\n\nExample of MOST PEAKED distribution:")
    print("-" * 80)
    most_peaked_idx = np.argmax(probs)
    example_logits = all_logits[most_peaked_idx]
    example_state = all_states[most_peaked_idx]

    # Get valid actions for this state
    valid_mask = np.isfinite(example_logits)
    valid_logits = example_logits[valid_mask]
    valid_actions = np.where(valid_mask)[0]

    print(f"State: {example_state}")
    print(f"Valid actions: {valid_actions}")
    print(f"Logits: {valid_logits}")

    # Compute softmax
    p = np.exp(valid_logits) / np.exp(valid_logits).sum()
    sorted_idx = np.argsort(p)[::-1]

    print(f"\nAction probabilities (sorted):")
    for i in sorted_idx[:min(5, len(sorted_idx))]:
        print(f"  Action {valid_actions[i]}: logit={valid_logits[i]:6.2f}, prob={p[i]:.6f}")

    # Simulate nucleus sampling
    cumsum = np.cumsum(p[sorted_idx])
    cutoff = np.where(cumsum > 0.9)[0]
    if len(cutoff) > 0:
        cutoff = cutoff[0] + 1
    else:
        cutoff = len(p)

    print(f"\nNucleus sampling (top_p=0.9):")
    print(f"  Nucleus size: {cutoff} actions")
    print(f"  Actions in nucleus: {valid_actions[sorted_idx[:cutoff]]}")
    if cutoff == 1:
        print(f"  WARNING: Nucleus collapsed to single action (greedy)!")


if __name__ == "__main__":
    # Inspect top_p model (fails with MCE=0)
    print("\n" + "=" * 80)
    print("1. TOP-P MODEL (MCE=0.0, mode collapse)")
    print("=" * 80)
    top_p_path = Path("results/ablations/sampling/top_p_seed42/checkpoint.pt")
    if top_p_path.exists():
        inspect_policy_logits(top_p_path, num_samples=20)
    else:
        print(f"Checkpoint not found: {top_p_path}")

    # Compare with categorical model (succeeds)
    print("\n\n" + "=" * 80)
    print("2. CATEGORICAL MODEL (MCE=0.18, diverse modes)")
    print("=" * 80)
    categorical_path = Path("results/ablations/sampling/categorical_seed42/checkpoint.pt")
    if categorical_path.exists():
        inspect_policy_logits(categorical_path, num_samples=20)
    else:
        print(f"Checkpoint not found: {categorical_path}")
