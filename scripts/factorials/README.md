# Factorial Experiments Runner

This directory contains the script for running factorial experiments that test interactions between multiple factors in Multi-Objective GFlowNets.

## Overview

Factorial experiments test **all combinations** of factor levels to discover **interaction effects**. This is different from ablation studies which test one factor at a time.

**Example**: Does the optimal sampling temperature depend on model capacity?
- Ablations would tell you: "Medium capacity is best" and "High temp is best"
- Factorial reveals: "Small models work best with low temp, large models need high temp"

## Files

- `run_factorial_experiment.py` - Main execution script
- `../configs/factorials/capacity_sampling_2way.yaml` - Capacity × Temperature factorial
- `../configs/factorials/sampling_loss_2way.yaml` - Temperature × Loss factorial

## Usage

### Basic Usage

```bash
python scripts/factorials/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --output_dir results/factorials/capacity_sampling
```

### Dry Run (Preview)

See what would be run without actually executing:

```bash
python scripts/factorials/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --dry-run
```

Output:
```
================================================================================
DRY RUN - Would execute the following experiments:
================================================================================

    1. [RUN] small_low_seed42
       Condition: small_low
       Factors: capacity=small, temperature=low
       Seed: 42

    2. [RUN] small_high_seed42
       Condition: small_high
       Factors: capacity=small, temperature=high
       Seed: 42
...
Total to run: 45
```

### Resume from Interruption

If experiments are interrupted, resume from where you left off:

```bash
python scripts/factorials/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --resume
```

The script will:
- Load existing `results.csv` or `results_temp.csv`
- Skip already completed experiments
- Continue from the next uncompleted experiment

### Run Specific Conditions

Test specific factor combinations only:

```bash
python scripts/factorials/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --conditions small_low,medium_high,large_veryhigh
```

### GPU Support

Use CUDA for faster training:

```bash
python scripts/factorials/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --device cuda
```

## Complete Examples

### Capacity × Sampling Factorial (Week 7)

```bash
# Full run (45 experiments, ~18 hours with 10 parallel jobs)
python scripts/factorials/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --output_dir results/factorials/capacity_sampling_2way
```

**Design**: 3 capacity levels × 3 temperature levels × 5 seeds = 45 runs

**Research Question**: Does optimal sampling temperature depend on model capacity?

**Expected Interaction**: Small models prefer lower temperature, large models need higher temperature to leverage capacity.

### Sampling × Loss Factorial (Week 8)

```bash
# Full run (45 experiments)
python scripts/factorials/run_factorial_experiment.py \
    --config configs/factorials/sampling_loss_2way.yaml \
    --output_dir results/factorials/sampling_loss_2way
```

**Design**: 3 temperature levels × 3 loss types × 5 seeds = 45 runs

**Research Question**: Does optimal loss function depend on exploration strategy?

**Expected Interaction**: Better credit assignment (SubTB) may require less exploration.

## Output Structure

```
results/factorials/capacity_sampling_2way/
├── experiment_config.yaml        # Full configuration used
├── results.csv                   # Final aggregated results
├── results_temp.csv              # Incremental results (during run)
├── failed.json                   # Failed experiments (if any)
├── small_low_seed42/
│   ├── config.json              # Experiment-specific config
│   ├── metrics.json             # All computed metrics
│   ├── checkpoint.pt            # Model checkpoint
│   ├── objectives.npy           # Evaluated objectives
│   ├── preferences.npy          # Evaluated preferences
│   └── training_history.json   # Training loss curve
├── small_low_seed153/
│   └── ...
└── ...
```

## Results Analysis

After experiments complete, analyze results:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/factorials/capacity_sampling_2way/results.csv')

# Create interaction plot
fig, ax = plt.subplots(figsize=(10, 6))
for temp in df['temperature_level'].unique():
    subset = df[df['temperature_level'] == temp]
    means = subset.groupby('capacity_level')['mce'].mean()
    ax.plot(means.index, means.values, 'o-', label=f'temp={temp}', linewidth=2)

ax.set_xlabel('Capacity')
ax.set_ylabel('Mode Coverage Entropy (MCE)')
ax.set_title('Capacity × Temperature Interaction')
ax.legend()
plt.tight_layout()
plt.savefig('interaction_plot.pdf')
```

**Interpreting Interaction Plots**:
- **Parallel lines** = No interaction (factors are independent)
- **Non-parallel lines** = Interaction exists (optimal settings depend on context)
- **Crossing lines** = Strong interaction (winner reverses based on other factor)

## Statistical Analysis

Run two-way ANOVA to test for interactions:

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Two-way ANOVA
model = ols('mce ~ C(capacity_level) + C(temperature_level) + C(capacity_level):C(temperature_level)',
            data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)
```

**Key Results**:
- **Main effect of capacity**: F-statistic, p-value
- **Main effect of temperature**: F-statistic, p-value
- **Interaction effect**: F-statistic, p-value (this is what we care about!)

If interaction p < 0.05: **Factors depend on each other!**

## Timeline and Budget

### Capacity × Sampling Factorial (Week 7)
- **Runs**: 45 (9 conditions × 5 seeds)
- **Time per run**: ~24 minutes
- **Total sequential**: 18 hours
- **Total parallel** (10 jobs): ~1.8 hours

### Sampling × Loss Factorial (Week 8)
- **Runs**: 45 (9 conditions × 5 seeds)
- **Time per run**: ~24 minutes
- **Total sequential**: 18 hours
- **Total parallel** (10 jobs): ~1.8 hours

### Combined
- **Total runs**: 90
- **Total time**: ~3.6 hours (with 10 parallel jobs)

## Troubleshooting

### Import Errors

If you get import errors:
```
ImportError: No module named 'src.models.mogfn_pc'
```

Make sure you're running from the project root:
```bash
cd /path/to/diversity-mogfn
python scripts/factorials/run_factorial_experiment.py --config configs/...
```

### CUDA Out of Memory

If GPU runs out of memory, reduce batch size in the config YAML:
```yaml
fixed:
  batch_size: 64  # Reduced from 128
```

Or use CPU:
```bash
python scripts/factorials/run_factorial_experiment.py \
    --config configs/... \
    --device cpu
```

### Experiments Failing

Check the `failed.json` file for error messages:
```bash
cat results/factorials/capacity_sampling_2way/failed.json
```

Common issues:
- Invalid configuration parameters
- Missing dependencies
- Memory issues

## Next Steps

After completing factorial experiments:

1. **Analyze Results** (Week 9)
   - Run statistical tests (ANOVA)
   - Create interaction plots
   - Identify best 2-3 configurations

2. **Validation Phase** (Weeks 10-12)
   - Test selected configs on real tasks
   - 3-grams (discrete sequences)
   - Molecules (drug discovery)
   - Sequences (proteins/DNA)

3. **Transfer Analysis** (Week 13)
   - Compare rankings across tasks
   - Measure transfer strength
   - Create practical guidelines

## References

- Configuration files: `configs/factorials/`
- Experimental strategy: `EXPERIMENTAL_STRATEGY.md`
- Experimental flow: `EXPERIMENTAL_FLOW.md`
- Factorial design guide: `configs/factorials/README.md`
