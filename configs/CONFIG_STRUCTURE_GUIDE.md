# GFlowNet Config Structure Guide

## Correct Configuration Structure

Use `configs/factorials/sampling_loss_2way.yaml` as the reference template for all future configs.

### ✅ CORRECT Parameter Names

```yaml
fixed:
  # Task configuration
  task: "hypergrid"              # ✓ NOT "environment"
  grid_size: [32, 32]            # ✓ List format, NOT single value 32

  # Model architecture
  hidden_dim: 128
  num_layers: 4
  conditioning: "concat"         # Options: "concat", "film"
  activation: "relu"

  # Preference sampling
  preference_distribution: "dirichlet"
  dirichlet_alpha: 1.5
  num_preferences_per_batch: 16
  sampling_strategy: "categorical"

  # Training
  max_iterations: 4000
  batch_size: 128
  optimizer: "adam"
  learning_rate: 0.001
  gradient_clip: 10.0

  # Evaluation
  eval_every: 500
  eval_samples: 1000
  final_eval_samples: 10000

  # Seeds
  num_seeds: 5
  base_seed: 42

factors:
  temperature:
    levels:
      low:
        temperature: 1.0        # ✓ "temperature", NOT "temperature_sampling"
        label: "Low (τ=1.0)"

  offpolicy:
    levels:
      off0:
        off_policy_ratio: 0.0   # ✓ "off_policy_ratio"
        label: "On-policy (ε=0.0)"

  loss:
    levels:
      subtb_entropy:
        loss_function: "subtrajectory_balance"
        loss_params:
          lambda_: 0.9
          log_reward_clip: 10.0
        regularization: "entropy"
        regularization_params:
          beta: 0.01

conditions:
  - name: "low_off0_subtb"
    temperature: "low"          # ✓ References factor level name
    offpolicy: "off0"
    loss: "subtb_entropy"
```

### ❌ BROKEN Parameter Names (DO NOT USE)

```yaml
fixed:
  environment: "hypergrid"      # ❌ WRONG - causes mode collapse
  grid_size: 32                 # ❌ WRONG - should be [32, 32]
  temperature_sampling: 5.0     # ❌ WRONG - should be in factor levels
  num_objectives: 2             # ❌ UNNECESSARY - remove
  modifications: "none"         # ❌ UNNECESSARY - remove
  off_policy_ratio: 0.0         # ❌ WRONG - should be in factor levels

factors:
  temperature:
    levels:
      temp5:
        temperature_sampling: 5.0  # ❌ WRONG - use "temperature"
```

## Task-Specific Configurations

### HyperGrid
```yaml
fixed:
  task: "hypergrid"
  grid_size: [32, 32]
```

### Molecules
```yaml
fixed:
  environment: 'molecules'      # ✓ Molecules uses "environment"
  max_fragments: 8
  num_fragments_library: 15
  use_rdkit: True
  objective_properties:
    - "qed"
    - "sa"
    - "logp"
```

### Sequences
```yaml
fixed:
  environment: 'sequences'      # ✓ Sequences uses "environment"
  seq_length: 20
  temperature_seq: 37.0         # ✓ This is RNA folding temp, not sampling
  use_viennarna: True
  objective_properties:
    - "free_energy"
    - "num_base_pairs"
    - "inverse_length"
```

### N-grams
```yaml
fixed:
  task: "ngrams"                # ✓ N-grams uses "task"
  vocab_size: 4
  seq_length: 8
  ngram_length: 2
  normalize_rewards: true
```

## Common Mistakes and Fixes

### Issue 1: Universal Mode Collapse (MCE=0 for all conditions)

**Symptom**: All experiments show MCE=0, num_modes=1, regardless of parameters

**Cause**: Using broken config structure with wrong parameter names

**Fix**:
1. Change `environment: "hypergrid"` → `task: "hypergrid"`
2. Change `grid_size: 32` → `grid_size: [32, 32]`
3. Change `temperature_sampling` → `temperature` (in factor levels)
4. Remove `num_objectives`, `modifications`, `modifications_params`

### Issue 2: Config Generated but Experiments Don't Match

**Symptom**: YAML file looks correct but experiments use old broken config

**Cause**: Cached config.json files in experiment directories

**Fix**:
```bash
# Delete results and re-run
rm -rf results/validation/temp_offpolicy/
python3 scripts/factorials/hypergrid/run_factorial_hypergrid.py \
    --config configs/validation/temp_offpolicy_interaction.yaml \
    --output_dir results/validation/temp_offpolicy
```

### Issue 3: Mode Collapse at Low Temperature

**Symptom**: MCE=0 only for temperature=1.0 conditions

**Cause**: This is EXPECTED BEHAVIOR, not a bug!

**Explanation**: Low temperature (τ=1.0) makes sampling too greedy on HyperGrid, causing exploitation of first solution found. This is a real phenomenon, not a configuration error.

**Solutions**:
- Use higher temperature (τ=2.0 or τ=5.0) for better diversity
- Add off-policy exploration to rescue low temperature
- Use entropy regularization

## Temperature Effects on HyperGrid

Based on factorial study results:

| Temperature | Expected MCE | Behavior |
|-------------|--------------|----------|
| τ=1.0 (low) | ~0.00 | COLLAPSE - too greedy, exploits first solution |
| τ=2.0 (high) | ~0.36 | HEALTHY - good exploration/exploitation balance |
| τ=5.0 (very high) | ~0.37 | HEALTHY - high exploration, may be unstable with off-policy |

## Checklist for New Configs

Before running a new config:

- [ ] Use correct task identifier (`task` for HyperGrid/N-grams, `environment` for molecules/sequences)
- [ ] Use list format for `grid_size: [32, 32]`
- [ ] Use `temperature` in factor levels (NOT `temperature_sampling`)
- [ ] Use `off_policy_ratio` in factor levels (NOT in `fixed`)
- [ ] Remove unnecessary parameters: `num_objectives`, `modifications`
- [ ] Test with quick config (500 iterations, 2 seeds) first
- [ ] Check that generated `config.json` matches YAML structure

## Reference Configs

**Working configs to copy from:**
- `configs/factorials/sampling_loss_2way.yaml` - HyperGrid 2-way factorial ✅
- `configs/factorials/capacity_loss_2way.yaml` - Capacity × Loss ✅

**Fixed validation configs:**
- `configs/validation/temp_offpolicy_interaction.yaml` - Temperature × Off-policy ✅
- `configs/validation/conditioning_loss_interaction.yaml` - Conditioning × Loss ✅

**Test configs:**
- `tests/validation/test_temp_offpolicy_config.yaml` - Quick test (500 iter) ✅
- `tests/validation/test_conditioning_loss_config.yaml` - Quick test (500 iter) ✅
