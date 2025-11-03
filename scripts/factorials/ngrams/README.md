# N-grams Factorial Experiments

Factorial experiments for Multi-Objective GFlowNets on the N-grams sequence generation environment.

## Overview

The N-grams environment generates sequences of fixed length from a vocabulary, where objectives are counts of different n-gram patterns (e.g., "AA", "BB", "AB", "BA" for bigrams).

## Supported Factorial Configurations

### 1. Capacity × Sampling
Tests interaction between model capacity and sampling temperature.

### 2. Capacity × Loss
Tests interaction between model capacity and loss function.

### 3. Sampling × Loss
Tests interaction between sampling temperature and loss function.

## Usage

### Basic Commands

```bash
# Activate environment
source .venv/bin/activate

# Dry run (preview without running)
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --dry-run

# Run full experiment
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --output_dir results/factorials/ngrams_capacity_sampling

# Resume interrupted experiment
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --resume

# Run specific conditions only
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --conditions small_low,medium_high
```

## Configuration Parameters

The script uses sensible defaults for parameters not specified in the config:

### Environment Parameters
- `vocab_size`: 4 (default: A, B, C, D)
- `seq_length`: 8 (length of generated sequences)
- `ngram_length`: 2 (bigrams by default)
- `objective_patterns`: None (auto-generated: AA, BB, AB, BA for bigrams)
- `normalize_rewards`: True (normalize counts by max possible)

### Model Parameters
- `temperature`: 1.0 (sampling temperature)
- `sampling_strategy`: 'categorical' (sampling method)
- `conditioning`: 'concat' (preference conditioning type)
- `preference_distribution`: 'dirichlet' (preference sampling)
- `dirichlet_alpha`: 1.5 (Dirichlet concentration)

### Training Parameters
- `learning_rate`: 0.001
- `loss_function`: 'trajectory_balance'
- `gradient_clip`: 10.0
- `max_iterations`: 5000
- `batch_size`: 128
- `num_preferences_per_batch`: 16
- `final_eval_samples`: 10000

## Output Structure

```
results/factorials/{experiment_name}/
├── {condition}_seed{seed}/
│   ├── config.json              # Full experiment configuration
│   ├── metrics.json             # All computed metrics
│   ├── objectives.npy           # Final objectives (N×num_objectives)
│   ├── preferences.npy          # Sampled preferences (N×num_objectives)
│   ├── checkpoint.pt            # Model checkpoint
│   └── training_history.json    # Training loss history
├── results.csv                  # Combined results from all experiments
└── experiment_config.yaml       # Copy of the factorial config
```

## Differences from HyperGrid

The N-grams script differs from the HyperGrid version in:

1. **Environment**: Uses `NGrams` instead of `HyperGrid`
   ```python
   env = NGrams(
       vocab_size=4,
       seq_length=8,
       ngram_length=2,
       objective_patterns=['AA', 'BB', 'AB', 'BA']
   )
   ```

2. **Reference Point**: Depends on normalization
   ```python
   if env.normalize_rewards:
       reference_point = [1.1] * env.num_objectives
   else:
       reference_point = [env.max_count + 1.0] * env.num_objectives
   ```

3. **State Dimension**: Variable based on sequence length
   - HyperGrid: Fixed based on grid size (e.g., 2 for 32×32)
   - N-grams: 1 + seq_length (position + sequence)

4. **Action Space**: Vocabulary + DONE action
   - HyperGrid: 4 actions (up, down, left, right)
   - N-grams: vocab_size + 1 actions (0 to vocab_size-1 characters + DONE)

## Example: Creating a New N-grams Factorial Config

```yaml
experiment_name: "ngrams_capacity_sampling_2way"
study_type: "factorial"

fixed:
  task: "ngrams"
  vocab_size: 4
  seq_length: 8
  ngram_length: 2
  normalize_rewards: true

  # Loss function
  loss_function: "subtrajectory_balance"
  loss_params:
    lambda_: 0.9
    log_reward_clip: 10.0

  # Training
  max_iterations: 4000
  batch_size: 128
  learning_rate: 0.001

  # Seeds
  num_seeds: 5
  base_seed: 42

factors:
  capacity:
    description: "Model size"
    levels:
      small:
        hidden_dim: 64
        num_layers: 2
      large:
        hidden_dim: 256
        num_layers: 4

  temperature:
    description: "Sampling temperature"
    levels:
      low:
        temperature: 1.0
      high:
        temperature: 2.0

# Auto-generates all combinations: small_low, small_high, large_low, large_high
```

## Tips

1. **Start with dry run**: Always use `--dry-run` first to preview what will be executed

2. **Test single condition**: Use `--conditions` to test one condition before running the full factorial
   ```bash
   python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
       --config your_config.yaml \
       --conditions small_low
   ```

3. **Resume capability**: The script automatically saves progress. Use `--resume` to continue after interruptions

4. **Mode collapse**: N-grams may be more prone to mode collapse than HyperGrid due to discrete action space. Monitor `num_modes` in metrics.

5. **Sequence length**: Longer sequences increase state space exponentially. Start with `seq_length=6` or `8` for testing.

## Troubleshooting

### All objectives are zero
- Check that `normalize_rewards=True` in config
- Verify n-gram patterns exist in generated sequences

### Mode collapse (all identical sequences)
- Increase entropy regularization
- Increase sampling temperature
- Increase model capacity
- Use the test factorial script to identify which configurations work

### Out of memory
- Reduce `batch_size`
- Reduce `final_eval_samples`
- Reduce model capacity
- Use smaller `seq_length`

## Related Scripts

- `scripts/factorials/hypergrid/run_factorial_experiment.py` - HyperGrid version
- `tests/factorials/test_capacity_sampling_factorial.py` - Quick test script
- `scripts/run_ablation_study.py` - Single-factor ablation studies
