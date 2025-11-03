# N-grams Factorial Experiment Configurations

Three factorial experiment configurations for the N-grams sequence generation environment.

## Created Configurations

### 1. `ngrams_capacity_sampling_2way.yaml` (10KB)
**Factors**: Model Capacity × Sampling Temperature
**Design**: 3 × 3 = 9 conditions × 5 seeds = 45 runs

| Capacity | Temperature | Combinations |
|----------|-------------|--------------|
| small (64×2) | low (τ=1.0) | small_low |
| medium (128×4) | high (τ=2.0) | medium_high |
| large (256×6) | very_high (τ=5.0) | large_veryhigh |

**Research Question**: Does optimal sampling strategy depend on model capacity for discrete sequence generation?

### 2. `ngrams_capacity_loss_2way.yaml` (9.3KB)
**Factors**: Model Capacity × Loss Function
**Design**: 3 × 3 = 9 conditions × 5 seeds = 45 runs

| Capacity | Loss Function | Combinations |
|----------|---------------|--------------|
| small (64×2) | TB | small_tb |
| medium (128×4) | SubTB(λ=0.9) | medium_subtb |
| large (256×6) | SubTB+Entropy | large_subtb_entropy |

**Research Question**: Does optimal loss function depend on model capacity for discrete sequence generation?

### 3. `ngrams_sampling_loss_2way.yaml` (10KB)
**Factors**: Sampling Temperature × Loss Function
**Design**: 3 × 3 = 9 conditions × 5 seeds = 45 runs
**Fixed**: Medium capacity (128×4)

| Temperature | Loss Function | Combinations |
|-------------|---------------|--------------|
| low (τ=1.0) | TB | low_tb |
| high (τ=2.0) | SubTB(λ=0.9) | high_subtb |
| very_high (τ=5.0) | SubTB+Entropy | veryhigh_subtb_entropy |

**Research Question**: Does optimal loss function depend on exploration strategy?

## Quick Start

### 1. Dry Run (Preview Experiments)
```bash
# Preview capacity × sampling
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --dry-run

# Preview capacity × loss
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_loss_2way.yaml \
    --dry-run

# Preview sampling × loss
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_sampling_loss_2way.yaml \
    --dry-run
```

### 2. Test Single Condition (Quick Verification)
```bash
# Test one condition to verify setup
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --conditions medium_high \
    --output_dir results/factorials/test_ngrams
```

### 3. Run Full Experiments
```bash
# Run capacity × sampling factorial
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --output_dir results/factorials/ngrams_capacity_sampling

# Run capacity × loss factorial
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_loss_2way.yaml \
    --output_dir results/factorials/ngrams_capacity_loss

# Run sampling × loss factorial
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_sampling_loss_2way.yaml \
    --output_dir results/factorials/ngrams_sampling_loss
```

### 4. Resume Interrupted Experiments
```bash
# Automatically skips completed experiments
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml \
    --resume
```

## Configuration Details

### Common Parameters (All Configs)

**Environment**:
- `vocab_size: 4` (A, B, C, D)
- `seq_length: 8` (8-character sequences)
- `ngram_length: 2` (bigrams)
- `normalize_rewards: true` (counts normalized by max)
- `objective_patterns: ['AA', 'BB', 'AB', 'BA']` (auto-generated)

**Training**:
- `max_iterations: 8000` (increased from 4000 for proper convergence)
- `batch_size: 128`
- `learning_rate: 0.001`
- `gradient_clip: 10.0`
- `num_seeds: 5` (42, 153, 264, 375, 486)

**Evaluation**:
- `eval_every: 500`
- `final_eval_samples: 10000`

### Variable Parameters

| Config | Factor A | Factor B |
|--------|----------|----------|
| capacity_sampling | Capacity (small/medium/large) | Temperature (1.0/2.0/5.0) |
| capacity_loss | Capacity (small/medium/large) | Loss (TB/SubTB/SubTB+Entropy) |
| sampling_loss | Temperature (1.0/2.0/5.0) | Loss (TB/SubTB/SubTB+Entropy) |

## Expected Computational Resources

**Per Experiment**:
- Runtime: ~30 minutes (8000 iterations)
- Storage: ~30 MB
- Memory: ~2 GB

**Full Factorial** (45 runs):
- Sequential: ~23 hours
- Parallel (10 jobs): ~2.3 hours
- Total storage: ~1.4 GB per factorial

## Key Differences from HyperGrid Configs

1. **Environment Setup**:
   ```yaml
   # N-grams
   task: "ngrams"
   vocab_size: 4
   seq_length: 8
   ngram_length: 2

   # vs HyperGrid
   task: "hypergrid"
   grid_size: [32, 32]
   ```

2. **Model Capacity**:
   - N-grams: Smaller models (64×2, 128×4, 256×6)
   - HyperGrid: Larger models (32×2, 128×4, 256×6)
   - Reason: N-grams state_dim = 9 vs HyperGrid state_dim = 2

3. **Training Iterations**:
   - N-grams: 8000 iterations (64x larger state space requires more)
   - HyperGrid: 4000 iterations
   - Reason: N-grams has 65,536 possible sequences vs 1,024 HyperGrid positions

4. **Expected Runtime**:
   - N-grams: ~30 min/experiment (8000 iterations)
   - HyperGrid: ~24 min/experiment (4000 iterations)

5. **Expected Diversity**:
   - N-grams: Lower MCE expected (more constrained space)
   - N-grams: Higher TDS expected (more path choices)

## Hypotheses Unique to N-grams

1. **Stronger Temperature Effects**: Discrete action space should show larger temperature effects
2. **SubTB Advantage**: Longer fixed-length trajectories benefit more from better credit assignment
3. **Entropy Importance**: Discrete spaces may benefit more from entropy regularization
4. **Mode Collapse Risk**: Small models more likely to collapse on N-grams

## Output Structure

```
results/factorials/ngrams_{experiment_name}/
├── {condition}_seed{seed}/
│   ├── config.json              # Full configuration
│   ├── metrics.json             # All metrics
│   ├── objectives.npy           # Final n-gram counts (10000×4)
│   ├── preferences.npy          # Sampled preferences (10000×4)
│   ├── checkpoint.pt            # Model checkpoint
│   └── training_history.json    # Training loss
├── results.csv                  # Combined results
└── experiment_config.yaml       # Experiment setup
```

## Troubleshooting

### Mode Collapse
**Symptom**: All objectives identical, MCE = 0
**Solution**:
- Use higher capacity (medium or large)
- Increase temperature
- Add entropy regularization

### Low Diversity
**Symptom**: MCE < 0.15, few unique sequences
**Solution**:
- Check if model converged (final_loss < 1.0)
- Try SubTB+Entropy loss
- Increase temperature to 2.0

### Out of Memory
**Solution**:
- Reduce batch_size to 64
- Reduce final_eval_samples to 5000
- Use smaller model capacity

## Comparison with HyperGrid

To compare N-grams vs HyperGrid results:

```bash
# Run both environments with same factorial
python scripts/factorials/hypergrid/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml

python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_sampling_2way.yaml

# Compare results
python scripts/compare_environments.py \
    --hypergrid results/factorials/capacity_sampling/results.csv \
    --ngrams results/factorials/ngrams_capacity_sampling/results.csv
```

## Next Steps

1. **Test First**: Run single condition to verify setup
2. **Run All Three**: Execute all three factorials
3. **Analyze**: Use statistical tests to identify main effects and interactions
4. **Compare**: Compare with HyperGrid results to identify environment-specific effects

## References

- N-grams Environment: `src/environments/ngrams.py`
- Factorial Script: `scripts/factorials/ngrams/run_factorial_experiment_ngrams.py`
- HyperGrid Configs: `configs/factorials/*_2way.yaml` (for comparison)
