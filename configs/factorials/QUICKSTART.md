# Factorial Experiments - Quick Start

## Overview

Factorial experiments test **interactions between factors** that were studied independently in ablation studies.

**Why Factorial Experiments?**
- Ablation studies found best settings for each factor *independently*
- But factors may **interact**: optimal setting for Factor A may depend on Factor B
- Factorial designs test all combinations to discover these interactions

## Available Configurations

### 1. **capacity_sampling_2way.yaml** (Recommended First)
Tests: Capacity × Temperature interaction

**Research Question**: Does optimal temperature depend on model size?

**Design**: 3 × 3 = 9 conditions
- Capacity: small, medium, large
- Temperature: 1.0, 2.0, 5.0
- Total: 45 runs (9 conditions × 5 seeds)

**Expected Finding**: Small models may prefer lower temperature (limited capacity), large models benefit from higher temperature.

### 2. **sampling_loss_2way.yaml** (Recommended Second)
Tests: Temperature × Loss function interaction

**Research Question**: Does optimal loss depend on exploration strategy?

**Design**: 3 × 3 = 9 conditions
- Temperature: 1.0, 2.0, 5.0
- Loss: TB, SubTB(0.9), SubTB+Entropy
- Total: 45 runs

**Expected Finding**: Better credit assignment (SubTB) may reduce need for high exploration.

### 3. **template_factorial.yaml**
Template for creating your own factorial experiments.

## Quick Usage

### Option 1: Use Existing Factorial Script (To Be Created)

```bash
# Run a 2-way factorial
python scripts/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --output_dir results/factorials/capacity_sampling
```

### Option 2: Adapt Ablation Script

```bash
# Modify run_ablation_study.py to handle factorial configs
python scripts/run_ablation_study.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --output_dir results/factorials/capacity_sampling
```

### Option 3: Manual Execution

Create individual experiment configs for each condition and run them:

```python
import yaml

# Load factorial config
with open('configs/factorials/capacity_sampling_2way.yaml') as f:
    config = yaml.safe_load(f)

# Generate all 9 combinations
conditions = config['conditions']
for condition in conditions:
    # Extract factor levels for this condition
    capacity_level = condition['capacity']
    temp_level = condition['temperature']

    # Get parameters from factor definitions
    capacity_params = config['factors']['capacity']['levels'][capacity_level]
    temp_params = config['factors']['temperature']['levels'][temp_level]

    # Create experiment config
    exp_config = {**config['fixed'], **capacity_params, **temp_params}

    # Run experiment...
```

## Understanding Factorial Results

### 1. Main Effects
How each factor affects outcomes **independently** (averaged across other factor).

```python
import pandas as pd
import seaborn as sns

df = pd.read_csv('results/factorials/capacity_sampling/results.csv')

# Main effect of capacity
df.groupby('capacity')['mce'].mean()

# Main effect of temperature
df.groupby('temperature')['mce'].mean()
```

### 2. Interaction Effects
How factors affect each other (non-additive effects).

**No Interaction** (Parallel lines):
```
       τ=1.0    τ=2.0    τ=5.0
Small   0.15     0.25     0.35   (+0.10 each step)
Medium  0.25     0.35     0.45   (+0.10 each step)
Large   0.35     0.45     0.55   (+0.10 each step)
```
→ Effect of temperature is **same** for all capacities

**Interaction** (Non-parallel lines):
```
       τ=1.0    τ=2.0    τ=5.0
Small   0.15     0.18     0.20   (+0.03, +0.02) small benefit
Medium  0.25     0.35     0.45   (+0.10, +0.10) moderate benefit
Large   0.35     0.55     0.65   (+0.20, +0.10) large benefit!
```
→ Effect of temperature **depends on** capacity

### 3. Visualization

**Interaction Plot**:
```python
import matplotlib.pyplot as plt

# If lines are parallel → no interaction
# If lines cross or diverge → interaction exists
sns.pointplot(data=df, x='capacity', y='mce', hue='temperature')
plt.title('Interaction Plot: Parallel = No Interaction')
plt.show()
```

**Heatmap**:
```python
pivot = df.pivot_table(values='mce', index='capacity', columns='temperature')
sns.heatmap(pivot, annot=True, fmt='.3f')
plt.title('MCE by Capacity × Temperature')
plt.show()
```

### 4. Statistical Test (Two-Way ANOVA)

```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Fit model with interaction term
model = ols('mce ~ C(capacity) + C(temperature) + C(capacity):C(temperature)',
            data=df).fit()

# ANOVA table
anova_results = anova_lm(model, typ=2)
print(anova_results)

# Look for:
# - C(capacity): Main effect of capacity
# - C(temperature): Main effect of temperature
# - C(capacity):C(temperature): INTERACTION (key result!)
```

**Interpretation**:
- **p < 0.05 for interaction term** → Significant interaction found!
- **p > 0.05 for interaction term** → No interaction, factors are independent

## Execution Recommendations

### Week 1: Capacity × Sampling (Most Important)
This interaction is most likely because:
- Model capacity limits what can be learned
- Sampling strategy determines what is explored
- Small models can't leverage complex exploration

```bash
python scripts/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml
```

**Decision Point**: If interaction found, consider capacity when choosing temperature.

### Week 2: Sampling × Loss
This interaction tests whether better losses reduce exploration needs:

```bash
python scripts/run_factorial_experiment.py \
    --config configs/factorials/sampling_loss_2way.yaml
```

**Decision Point**: If SubTB works well at lower temps, use it for stability.

### Week 3+: Optional Extensions
- Add capacity × loss factorial if budget allows
- Test 3-way interaction (capacity × sampling × loss) with reduced levels

## Creating Custom Factorials

1. **Copy template**:
   ```bash
   cp configs/factorials/template_factorial.yaml \
      configs/factorials/my_factorial.yaml
   ```

2. **Define factors and levels**:
   ```yaml
   factors:
     my_factor:
       levels:
         level1: { param: value1 }
         level2: { param: value2 }
   ```

3. **List all combinations**:
   ```yaml
   conditions:
     - name: "level1_level2"
       my_factor: "level1"
       my_other_factor: "level2"
   ```

4. **Run**: Use factorial script or adapt ablation script

## Expected Computational Cost

### Capacity × Sampling (3×3)
- **Conditions**: 9
- **Seeds per condition**: 5
- **Total runs**: 45
- **Time per run**: ~24 min
- **Total time**: ~18 hours sequential, ~2 hours parallel (10 jobs)

### Sampling × Loss (3×3)
- **Conditions**: 9
- **Total runs**: 45
- **Total time**: ~18 hours sequential, ~2 hours parallel

### Both Factorials
- **Total runs**: 90
- **Total time**: ~36 hours sequential, ~4 hours parallel

## Key Takeaways

1. **Main Effects**: What we learned from ablations (already known)
2. **Interactions**: **New knowledge** - do factors depend on each other?
3. **Practical Value**: Guidelines for setting multiple hyperparameters together
4. **Paper Contribution**: First systematic study of factor interactions in MOGFNs

## Next Steps After Factorials

1. **If no interactions found**:
   - Use best level from each factor independently
   - Final configuration = winner from each ablation

2. **If interactions found**:
   - Report as key finding
   - Provide context-dependent recommendations
   - May need additional experiments to map interaction

3. **Final validation**:
   - Test selected configuration(s) on holdout tasks
   - Verify robustness across seeds
   - Compare to baselines

## Support

- **Detailed docs**: See `README.md` in this directory
- **Template**: Use `template_factorial.yaml` for custom experiments
- **Examples**: See `capacity_sampling_2way.yaml` and `sampling_loss_2way.yaml`
