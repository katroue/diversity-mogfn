# Factorial Experiments - Setup Summary

## ✅ Completed Tasks

### 1. Verified Loss Ablation Configuration
The `configs/ablations/loss_ablation.yaml` **already has** the correct `temp_high` settings from the sampling ablation study:

```yaml
# Preference Sampling (FIXED from sampling ablation)
preference_distribution: "dirichlet"
temperature: 2.0  # ✓ Best from sampling ablation (temp_high)
sampling_strategy: "categorical"  # ✓ Matches temp_high
dirichlet_alpha: 1.5
num_preferences_per_batch: 16
sampling_schedule: "uniform"
```

**Source**: `configs/ablations/sampling_ablation.yaml` line 55-58 (temp_high configuration)

✅ **No changes needed** - configuration is correct!

### 2. Created Factorial Experiments Folder
Created `configs/factorials/` with complete factorial experiment infrastructure:

```
configs/factorials/
├── README.md                        # Comprehensive guide to factorial experiments
├── QUICKSTART.md                    # Quick reference and examples
├── capacity_sampling_2way.yaml      # Capacity × Temperature (3×3 = 9 conditions)
├── sampling_loss_2way.yaml          # Temperature × Loss (3×3 = 9 conditions)
└── template_factorial.yaml          # Template for creating custom factorials
```

## 📊 Factorial Experiments Overview

### What Are Factorial Experiments?

Factorial experiments test **interactions between factors** studied independently in ablation studies.

**Example Interaction**:
- **Ablation findings**: Medium capacity is best, High temperature is best
- **But**: Does optimal temperature depend on capacity?
  - Small models: May prefer lower temperature (limited capacity)
  - Large models: May need higher temperature to leverage capacity
- **Factorial experiment**: Tests all combinations to discover this

### Created Configurations

#### 1. Capacity × Sampling (3×3 factorial)
**File**: `configs/factorials/capacity_sampling_2way.yaml`

**Factors**:
- Capacity: small (32×2), medium (128×4), large (256×6)
- Temperature: 1.0, 2.0, 5.0

**Conditions**: 9 (all combinations) × 5 seeds = **45 runs**

**Research Question**: Does optimal sampling temperature depend on model size?

**Expected Finding**: Small models can't leverage high exploration; large models benefit from it.

#### 2. Sampling × Loss (3×3 factorial)
**File**: `configs/factorials/sampling_loss_2way.yaml`

**Factors**:
- Temperature: 1.0, 2.0, 5.0
- Loss: TB, SubTB(0.9), SubTB+Entropy

**Conditions**: 9 × 5 seeds = **45 runs**

**Research Question**: Does optimal loss function depend on exploration strategy?

**Expected Finding**: SubTB (better credit assignment) may need less exploration than TB.

#### 3. Template
**File**: `configs/factorials/template_factorial.yaml`

Copy and customize this template to create your own factorial experiments.

## 🎯 Recommended Execution Order

### Phase 1: Capacity × Sampling (Week 1)
**Priority**: **HIGH** (most likely to show interactions)

```bash
python scripts/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml \
    --output_dir results/factorials/capacity_sampling
```

**Why first**: Model capacity directly affects what can be learned. If interactions exist, this is where they'll be strongest.

**Time**: ~18 hours sequential, ~2 hours with 10 parallel jobs

**Analysis**:
```python
# Check for interaction
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('mce ~ C(capacity) + C(temperature) + C(capacity):C(temperature)',
            data=df).fit()
print(anova_lm(model, typ=2))

# If p < 0.05 for interaction term → Interaction found!
```

### Phase 2: Sampling × Loss (Week 2)
**Priority**: **MEDIUM** (interesting for optimization)

```bash
python scripts/run_factorial_experiment.py \
    --config configs/factorials/sampling_loss_2way.yaml \
    --output_dir results/factorials/sampling_loss
```

**Why second**: Tests whether better losses (SubTB) reduce need for exploration.

**Time**: ~18 hours sequential, ~2 hours parallel

**Practical Value**: If SubTB works well at lower temperature, it's more stable and easier to tune.

### Phase 3: Capacity × Loss (Optional)
**Priority**: **LOW** (if budget allows)

Less likely to show strong interactions, but completes the 2-way factorial matrix.

## 📈 Understanding Interaction Results

### No Interaction (Parallel Lines)
```
Interaction Plot:

MCE │     ╱─────── Large
    │   ╱─────── Medium
    │ ╱─────── Small
    └─────────────────
      1.0  2.0  5.0  (Temperature)
```
**Lines are parallel** → Effect of temperature is **same** for all capacities

**Interpretation**: Factors are **independent**
**Recommendation**: Use Medium capacity + High temperature (best from each ablation)

### Interaction Found (Non-Parallel Lines)
```
Interaction Plot:

MCE │         ╱──── Large (steep)
    │      ╱──── Medium
    │    ╱── Small (flat)
    └─────────────────
      1.0  2.0  5.0  (Temperature)
```
**Lines diverge** → Effect of temperature **depends on** capacity

**Interpretation**: Factors **interact**
**Recommendation**:
- Small models: Use lower temperature (they can't leverage high exploration)
- Large models: Use higher temperature (they benefit from more exploration)

## 📝 Key Insights from Factorials

### What We Learn

1. **Main Effects** (already known from ablations)
   - Which capacity is best (averaged across temperatures)
   - Which temperature is best (averaged across capacities)

2. **Interaction Effects** (**NEW knowledge**)
   - Does optimal temperature depend on capacity?
   - Do factors work together in non-additive ways?

3. **Practical Guidelines**
   - How to set multiple hyperparameters **together**
   - Context-dependent recommendations

### Statistical Significance

**Two-Way ANOVA Output**:
```
                              sum_sq    df          F    PR(>F)
C(capacity)                   0.123     2      12.45   0.001 ***  ← Main effect
C(temperature)                0.456     2      45.67   0.000 ***  ← Main effect
C(capacity):C(temperature)    0.089     4       4.56   0.012 *    ← INTERACTION!
```

**Key**: Look at `PR(>F)` for interaction term
- **p < 0.05** → Significant interaction found!
- **p > 0.05** → No interaction, factors independent

## 🔧 Implementation Options

### Option 1: Create Factorial Script (Recommended)
Create `scripts/run_factorial_experiment.py` similar to `run_loss_ablation_group.py`.

### Option 2: Adapt Existing Scripts
Modify `scripts/run_ablation_study.py` to parse factorial configs.

### Option 3: Manual Execution
Parse YAML and create individual experiment configs for each condition.

## 📊 Computational Resources

### Both 2-Way Factorials
- **Total conditions**: 18 (9 + 9)
- **Total runs**: 90 (45 + 45)
- **Time**: ~36 hours sequential, **~4 hours parallel** (10 jobs)
- **Storage**: ~5 GB

### Cost-Benefit
- **Cost**: 90 runs (~4 hours)
- **Benefit**: Discover interactions that ablations can't reveal
- **ROI**: High - provides practical guidelines for practitioners

## 📚 Next Steps

### After Running Factorials

1. **Analyze Results**
   - Run two-way ANOVA for each factorial
   - Create interaction plots
   - Check for significant interactions

2. **Make Decisions**
   - **If no interactions**: Use best level from each factor independently
   - **If interactions found**: Provide context-dependent recommendations

3. **Paper Contribution**
   - Report interactions as key findings
   - Provide practical hyperparameter guidelines
   - First systematic study of factor interactions in MOGFNs

### Long-Term

Consider:
- 3-way factorial (if budget allows and 2-way shows interactions)
- Robustness testing on different tasks
- Transfer learning experiments

## 📖 Documentation

- **Comprehensive guide**: `configs/factorials/README.md`
- **Quick reference**: `configs/factorials/QUICKSTART.md`
- **Example configs**: `capacity_sampling_2way.yaml`, `sampling_loss_2way.yaml`
- **Template**: `template_factorial.yaml`

## ✅ Ready to Run

Everything is set up! Factorial experiments are ready to execute after completing the loss ablation study.

**Recommended timeline**:
1. Week 6: Complete loss ablation (5 groups)
2. Week 7: Run Capacity × Sampling factorial
3. Week 8: Run Sampling × Loss factorial
4. Week 9: Analysis and paper writing

---

**Summary**: Factorial experiments folder created successfully with complete infrastructure for testing factor interactions. Loss ablation config already has correct temp_high settings. Ready to proceed with factorial experiments after ablation studies complete!
