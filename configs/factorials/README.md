# Factorial Experiment Configurations

This directory contains configurations for factorial experiments that test interactions between multiple factors simultaneously.

## Background

After completing the ablation studies (capacity, sampling, loss), we identified the best settings for each factor independently. However, these factors may **interact** in non-trivial ways. Factorial experiments test all combinations to discover these interactions.

## Ablation Study Results (Winners)

From previous ablation studies, we identified:

### Capacity Ablation
- **Winner**: Large capacity (128 hidden dim, 4 layers, concat conditioning)
- **Alternative**: Very large capacity for complex tasks

### Sampling Ablation
- **Winner**: temp_high (temperature=2.0, categorical sampling)
- **Key insight**: Higher temperature increases exploration and diversity

### Loss Ablation
- **Expected winner**: SubTB(λ=0.9) + Entropy(β=0.01)
- **Alternative**: Trajectory Balance with moderate entropy

## Factorial Experiment Types

### 1. Two-Factor Factorial (2-Way)
Test interactions between **two factors** at a time.

Example: Capacity × Sampling
- Factor A (Capacity): [small, medium, large]
- Factor B (Temperature): [1.0, 2.0, 5.0]
- Total combinations: 3 × 3 = 9 configs

### 2. Three-Factor Factorial (3-Way)
Test interactions between **three factors**.

Example: Capacity × Sampling × Loss
- Factor A (Capacity): [small, medium, large]
- Factor B (Temperature): [1.0, 2.0]
- Factor C (Loss): [TB, SubTB]
- Total combinations: 3 × 2 × 2 = 12 configs

### 3. Full Factorial
Test **all factors** at **all levels**.

⚠️ Warning: Combinatorial explosion!
- 3 capacity levels × 4 temp levels × 3 loss levels = 36 configs
- With 5 seeds each = 180 total runs

### 4. Fractional Factorial
Test **subset of combinations** using statistical design.

Benefits:
- Reduce computational cost
- Still detect main effects and key interactions
- Requires careful experimental design

## Configuration Files

### Core Factorial Designs

- **`capacity_sampling_2way.yaml`** - Capacity × Sampling interaction
- **`capacity_loss_2way.yaml`** - Capacity × Loss interaction
- **`sampling_loss_2way.yaml`** - Sampling × Loss interaction
- **`three_factor_factorial.yaml`** - Capacity × Sampling × Loss
- **`full_factorial.yaml`** - All factors, all levels

### Specialized Designs

- **`fractional_factorial_resolution4.yaml`** - Resolution IV fractional design
- **`response_surface.yaml`** - Continuous factor optimization
- **`robustness_test.yaml`** - Test winner robustness across conditions

## Usage

```bash
# Run a 2-way factorial experiment
python scripts/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --output_dir results/factorials/capacity_sampling

# Run fractional factorial
python scripts/run_factorial_experiment.py \
    --config configs/factorials/fractional_factorial_resolution4.yaml \
    --output_dir results/factorials/fractional
```

## Analysis

Factorial experiments enable:

1. **Main Effects**: How each factor affects outcomes independently
2. **Interaction Effects**: How factors affect each other
3. **ANOVA**: Statistical significance testing
4. **Response Surfaces**: Optimal factor combinations

Example analysis:
```python
import pandas as pd
from scipy.stats import f_oneway

df = pd.read_csv('results/factorials/capacity_sampling/results.csv')

# Two-way ANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('mce ~ C(capacity) + C(temperature) + C(capacity):C(temperature)',
            data=df).fit()
anova_results = anova_lm(model, typ=2)
print(anova_results)
```

## Recommended Execution Order

1. **Week 1**: Run `capacity_sampling_2way.yaml`
   - Most likely to show interactions
   - Capacity affects model expressiveness
   - Sampling affects exploration

2. **Week 2**: Run `sampling_loss_2way.yaml`
   - Loss affects training dynamics
   - May interact with exploration strategy

3. **Week 3**: Run `capacity_loss_2way.yaml` (if budget allows)
   - Capacity may affect which loss works best

4. **Week 4**: Run `three_factor_factorial.yaml` (reduced levels)
   - Only if significant interactions found in 2-way experiments

5. **Week 5**: Run `robustness_test.yaml`
   - Verify winner is robust

## Expected Outcomes

### Likely Interactions

1. **Capacity × Temperature**
   - Small models may need lower temperature (less capacity to explore)
   - Large models may benefit from higher temperature

2. **Loss × Temperature**
   - Entropy regularization may interact with sampling temperature
   - Both affect exploration

3. **Capacity × Loss**
   - SubTB may require sufficient capacity to work well
   - Small models may prefer simpler losses (TB)

### Unlikely Interactions

- Conditioning method × most other factors (orthogonal concerns)
- Batch size × capacity (mostly independent)

## Files in This Directory

- `README.md` - This file
- `capacity_sampling_2way.yaml` - Capacity × Sampling factorial
- `capacity_loss_2way.yaml` - Capacity × Loss factorial
- `sampling_loss_2way.yaml` - Sampling × Loss factorial
- `three_factor_factorial.yaml` - 3-way factorial
- `full_factorial.yaml` - Full factorial design
- `fractional_factorial_resolution4.yaml` - Fractional factorial
- `template_factorial.yaml` - Template for creating new designs

## Statistical Power

For detecting interactions:
- **Minimum**: 3 replicates (seeds) per condition
- **Recommended**: 5 replicates for medium effect sizes
- **Ideal**: 10 replicates for small effect sizes

With 5 seeds and 9 conditions (3×3 design):
- Total runs: 45
- Can detect medium-to-large interactions (Cohen's f ≥ 0.25)
- Power ≈ 0.80 for α = 0.05

## References

- Box, G. E., Hunter, J. S., & Hunter, W. G. (2005). Statistics for experimenters.
- Montgomery, D. C. (2017). Design and analysis of experiments (9th ed.).
- Factorial experiments in ML: Bouthillier et al. (2021). "Accounting for Variance in Machine Learning Benchmarks"
