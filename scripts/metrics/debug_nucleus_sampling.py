"""
Debug script to analyze nucleus (top-p) sampling behavior in MOGFN-PC.

This script tests the nucleus sampling implementation to understand why it causes
mode collapse (MCE=0.0) in the sampling strategy ablation.
"""

import torch
import torch.nn.functional as F
import numpy as np


def nucleus_sampling_current(logits, top_p=0.9, temperature=1.0):
    """Current implementation from mogfn_pc.py lines 333-357"""
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff index where cumulative prob exceeds top_p
    cutoff_idx = torch.where(cumulative_probs > top_p)[0]
    if len(cutoff_idx) > 0:
        cutoff_idx = cutoff_idx[0].item() + 1
    else:
        cutoff_idx = len(sorted_probs)

    # Keep only top-p probability mass
    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_indices = sorted_indices[:cutoff_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()  # Renormalize

    action_idx = torch.multinomial(nucleus_probs, 1).squeeze()
    action = nucleus_indices[action_idx]

    return action, nucleus_probs, nucleus_indices


def test_nucleus_behavior():
    """Test nucleus sampling with different probability distributions"""

    print("=" * 80)
    print("NUCLEUS (TOP-P) SAMPLING ANALYSIS")
    print("=" * 80)

    # Test 1: Uniform-like distribution (good for exploration)
    print("\n1. UNIFORM-LIKE DISTRIBUTION (exploration scenario)")
    print("-" * 80)
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])
    probs = F.softmax(logits, dim=-1)
    print(f"Logits: {logits.numpy()}")
    print(f"Probs:  {probs.numpy()}")

    action, nucleus_probs, nucleus_indices = nucleus_sampling_current(logits, top_p=0.9)
    print(f"\nNucleus (p=0.9):")
    print(f"  Selected indices: {nucleus_indices.numpy()}")
    print(f"  Renormalized probs: {nucleus_probs.numpy()}")
    print(f"  Sampled action: {action.item()}")

    # Test 2: Peaked distribution (common after training)
    print("\n2. PEAKED DISTRIBUTION (after training convergence)")
    print("-" * 80)
    logits = torch.tensor([5.0, 2.0, 1.0, 0.5, 0.1])
    probs = F.softmax(logits, dim=-1)
    print(f"Logits: {logits.numpy()}")
    print(f"Probs:  {probs.numpy()}")

    action, nucleus_probs, nucleus_indices = nucleus_sampling_current(logits, top_p=0.9)
    print(f"\nNucleus (p=0.9):")
    print(f"  Selected indices: {nucleus_indices.numpy()}")
    print(f"  Renormalized probs: {nucleus_probs.numpy()}")
    print(f"  Sampled action: {action.item()}")

    # Test 3: Very peaked distribution (mode collapse scenario)
    print("\n3. VERY PEAKED DISTRIBUTION (mode collapse risk)")
    print("-" * 80)
    logits = torch.tensor([10.0, 1.0, 0.5, 0.1, 0.01])
    probs = F.softmax(logits, dim=-1)
    print(f"Logits: {logits.numpy()}")
    print(f"Probs:  {probs.numpy()}")

    action, nucleus_probs, nucleus_indices = nucleus_sampling_current(logits, top_p=0.9)
    print(f"\nNucleus (p=0.9):")
    print(f"  Selected indices: {nucleus_indices.numpy()}")
    print(f"  Renormalized probs: {nucleus_probs.numpy()}")
    print(f"  Sampled action: {action.item()}")
    print(f"  WARNING: Only {len(nucleus_indices)} actions in nucleus!")

    # Test 4: Compare different top_p values
    print("\n4. EFFECT OF TOP_P VALUE (very peaked distribution)")
    print("-" * 80)
    logits = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
    probs = F.softmax(logits, dim=-1)
    print(f"Logits: {logits.numpy()}")
    print(f"Probs:  {probs.numpy()}")

    for top_p_val in [0.5, 0.7, 0.9, 0.95, 0.99]:
        action, nucleus_probs, nucleus_indices = nucleus_sampling_current(logits, top_p=top_p_val)
        print(f"\n  p={top_p_val}: {len(nucleus_indices)} actions, indices={nucleus_indices.numpy()}")

    # Test 5: Monte Carlo simulation - check actual sampling distribution
    print("\n5. MONTE CARLO SAMPLING TEST (1000 samples)")
    print("-" * 80)
    logits = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
    probs = F.softmax(logits, dim=-1)
    print(f"True probs: {probs.numpy()}")

    samples = []
    for _ in range(1000):
        action, _, _ = nucleus_sampling_current(logits, top_p=0.9)
        samples.append(action.item())

    empirical_probs = np.bincount(samples, minlength=len(logits)) / len(samples)
    print(f"Empirical distribution (top_p=0.9):")
    for i, (true_p, emp_p) in enumerate(zip(probs.numpy(), empirical_probs)):
        print(f"  Action {i}: true={true_p:.4f}, empirical={emp_p:.4f}")

    # Test 6: Compare with categorical sampling
    print("\n6. COMPARISON: NUCLEUS vs CATEGORICAL")
    print("-" * 80)
    logits = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5])
    probs = F.softmax(logits, dim=-1)

    # Categorical samples
    categorical_samples = []
    for _ in range(1000):
        action = torch.multinomial(probs, 1).squeeze()
        categorical_samples.append(action.item())
    categorical_probs = np.bincount(categorical_samples, minlength=len(logits)) / len(categorical_samples)

    # Nucleus samples
    nucleus_samples = []
    for _ in range(1000):
        action, _, _ = nucleus_sampling_current(logits, top_p=0.9)
        nucleus_samples.append(action.item())
    nucleus_dist = np.bincount(nucleus_samples, minlength=len(logits)) / len(nucleus_samples)

    print("Action | True Prob | Categorical | Nucleus (p=0.9)")
    print("-" * 60)
    for i in range(len(logits)):
        print(f"  {i}    |   {probs[i]:.4f}   |   {categorical_probs[i]:.4f}    |    {nucleus_dist[i]:.4f}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print("""
The nucleus (top-p) sampling implementation has a CRITICAL ISSUE for GFlowNets:

PROBLEM: Deterministic mode collapse during training
---------------------------------------------------------
1. During training, the policy network learns to assign high logits to
   high-reward states (e.g., the top-right corner in HyperGrid).

2. As training progresses, the logit distribution becomes VERY PEAKED:
   - Top action: logit ~ 10.0, prob ~ 0.99998
   - Other actions: logits < 1.0, probs < 0.00001

3. With top_p=0.9, the nucleus contains ONLY 1 ACTION (the argmax).
   - cutoff_idx = 1 (cumulative_probs[0] > 0.9)
   - nucleus_probs = [1.0]
   - nucleus_indices = [argmax]

4. Result: Nucleus sampling becomes IDENTICAL to greedy sampling.
   - All trajectories sample the same high-reward path
   - No exploration of alternative modes
   - MCE = 0.0 (only 1 mode discovered)

ROOT CAUSE: GFlowNet trajectory balance loss + nucleus sampling interaction
---------------------------------------------------------------------------
Unlike language models (where nucleus prevents repetition within a sequence),
GFlowNets sample ENTIRE TRAJECTORIES during training. The TB loss directly
optimizes the policy to match reward distributions, creating very peaked logits.

Nucleus sampling REDUCES diversity instead of increasing it because:
- It removes low-probability actions (which might lead to alternative modes)
- It concentrates sampling on the already-dominant mode
- It provides NO exploration benefit (unlike ε-greedy or high temperature)

SOLUTION OPTIONS:
-----------------
1. Use categorical sampling with HIGH TEMPERATURE (temp=2.0-5.0)
   - Smooths the distribution: softmax(logits/temp)
   - Maintains exploration throughout training

2. Use top-k sampling with LARGE k (k=5-10)
   - Ensures multiple actions always in consideration
   - More robust to peaked distributions

3. Add off-policy exploration (off_policy_ratio=0.1-0.25)
   - Guarantees diversity through random sampling

4. DO NOT use nucleus sampling for GFlowNets
   - Designed for autoregressive generation, not flow-based sampling
   - Incompatible with trajectory balance objective
""")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    test_nucleus_behavior()