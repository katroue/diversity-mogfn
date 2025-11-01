# Loss Ablation Study - Quick Start Guide

## TL;DR

```bash
# See what groups are available
python scripts/run_loss_ablation_group.py --list

# Run a single group (recommended approach)
python scripts/run_loss_ablation_group.py --group base_loss_comparison

# Or run all groups with interactive pauses
./scripts/run_loss_ablation.sh

# Or run all groups automatically (no pauses)
./scripts/run_loss_ablation.sh --auto
```

## What is This Study?

The loss ablation study systematically tests how different loss functions and regularization techniques affect diversity in Multi-Objective GFlowNets (MOGFNs).

**5 Experiment Groups:**
1. **Base Loss Comparison** (30 runs) - Which core loss is best?
2. **Entropy Regularization** (25 runs) - How does entropy affect diversity?
3. **KL Regularization** (15 runs) - Alternative to entropy?
4. **SubTB + Entropy** (20 runs) - Best combination?
5. **Loss Modifications** (15 runs) - Advanced techniques?

**Total: 105 experiments** (21 configs × 5 seeds each)

## Three Ways to Run

### Option 1: Interactive Group-by-Group (Recommended)

**Best for**: Analyzing results between groups, resource-constrained environments

```bash
# Week 1: Base losses
python scripts/run_loss_ablation_group.py --group base_loss_comparison

# Analyze results, then Week 2: Entropy
python scripts/run_loss_ablation_group.py --group entropy_regularization

# Continue with remaining groups...
```

### Option 2: Automated Sequential with Pauses

**Best for**: Running entire study with analysis checkpoints

```bash
# Runs all groups, pauses between each for analysis
./scripts/run_loss_ablation.sh

# Preview without running
./scripts/run_loss_ablation.sh --dry-run

# Resume from interruption
./scripts/run_loss_ablation.sh --resume
```

### Option 3: Fully Automated

**Best for**: When you want to run everything overnight

```bash
# No pauses, runs all groups back-to-back
./scripts/run_loss_ablation.sh --auto --resume
```

## Recommended Workflow

### Day 1-2: Base Loss Comparison

```bash
# Preview what will run
python scripts/run_loss_ablation_group.py \
    --group base_loss_comparison \
    --dry-run

# Run the group
python scripts/run_loss_ablation_group.py \
    --group base_loss_comparison

# Analyze results
cd results/ablations/loss/base_loss_comparison
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
print('\\nMean metrics by loss type:')
print(df.groupby('name')[['hypervolume', 'tds', 'mce', 'pas']].mean())
print('\\nBest by QDS (Quality-Diversity Score):')
print(df.groupby('name')['qds'].mean().sort_values(ascending=False))
"
```

**Decision Point**: Which base loss performed best? Update Groups 2-5 if needed.

### Day 3: Entropy Regularization

```bash
python scripts/run_loss_ablation_group.py \
    --group entropy_regularization

# Analyze
cd results/ablations/loss/entropy_regularization
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
print('\\nEntropy regularization effects:')
print(df.groupby('name')[['hypervolume', 'tds', 'mce', 'pas']].mean())
"
```

**Decision Point**: What's the optimal β for entropy?

### Day 4: KL Regularization

```bash
python scripts/run_loss_ablation_group.py \
    --group kl_regularization
```

### Day 5: SubTB + Entropy Combinations

```bash
python scripts/run_loss_ablation_group.py \
    --group subtb_entropy_sweep
```

### Day 6: Loss Modifications

```bash
python scripts/run_loss_ablation_group.py \
    --group loss_modifications
```

## File Structure

```
results/ablations/loss/
├── base_loss_comparison/
│   ├── results.csv                          # Aggregated results
│   ├── group_config.yaml                    # Group configuration
│   ├── trajectory_balance_seed42_config.json
│   ├── trajectory_balance_seed42/
│   │   ├── metrics.json
│   │   ├── objectives.npy
│   │   ├── model.pt
│   │   └── training_history.json
│   └── ...
├── entropy_regularization/
│   └── ...
└── ...
```

## Quick Commands Reference

### List all groups
```bash
python scripts/run_loss_ablation_group.py --list
```

### Dry-run a group
```bash
python scripts/run_loss_ablation_group.py --group <name> --dry-run
```

### Run a group
```bash
python scripts/run_loss_ablation_group.py --group <name>
```

### Resume interrupted group
```bash
python scripts/run_loss_ablation_group.py --group <name> --resume
```

### Run all groups
```bash
./scripts/run_loss_ablation.sh           # With pauses
./scripts/run_loss_ablation.sh --auto    # No pauses
```

## Analyzing Results

### Quick Summary
```python
import pandas as pd
df = pd.read_csv('results/ablations/loss/<group>/results.csv')

# Group by configuration
summary = df.groupby('name')[['hypervolume', 'tds', 'mce', 'pas']].agg(['mean', 'std'])
print(summary)
```

### Find Best Configuration
```python
# By QDS (Quality-Diversity Score)
best_qds = df.groupby('name')['qds'].mean().sort_values(ascending=False)
print("Best by QDS:", best_qds.head())

# By specific metric
best_mce = df.groupby('name')['mce'].mean().sort_values(ascending=False)
print("Best by MCE:", best_mce.head())
```

### Compare Groups
```python
import pandas as pd
import matplotlib.pyplot as plt

groups = ['base_loss_comparison', 'entropy_regularization']
dfs = [pd.read_csv(f'results/ablations/loss/{g}/results.csv') for g in groups]

for df, group in zip(dfs, groups):
    df['group'] = group

combined = pd.concat(dfs)
combined.groupby(['group', 'name'])['mce'].mean().unstack().plot(kind='bar')
plt.title('MCE by Group and Configuration')
plt.show()
```

## Troubleshooting

### Group won't start
**Check configuration exists:**
```bash
ls configs/ablations/loss_ablation.yaml
```

### Experiments failing
**Check failed experiments:**
```bash
cat results/ablations/loss/<group>/failed.json
```

### Resume not working
**Verify results file:**
```bash
ls results/ablations/loss/<group>/results.csv
head results/ablations/loss/<group>/results.csv
```

### Out of memory
**Run with fewer parallel jobs** (modify script to reduce batch size)

## Expected Results

Based on hypotheses in `configs/ablations/loss_ablation.yaml`:

1. **Base Loss**: SubTB(λ=0.9) > TB > DB > FM for diversity
2. **Entropy**: β=0.05 optimal, increases MCE by 20-30%
3. **Too much regularization**: β≥0.5 hurts quality
4. **Best overall**: SubTB(0.9) + Entropy(β=0.05)

## Next Steps After Completion

1. **Aggregate results** across all groups
2. **Statistical analysis** (ANOVA, post-hoc tests)
3. **Create visualizations** (plots, tables)
4. **Select best configuration** for factorial experiments
5. **Document findings** for paper

## More Information

- **Detailed guide**: `scripts/LOSS_ABLATION_README.md`
- **Configuration**: `configs/ablations/loss_ablation.yaml`
- **Main script**: `scripts/run_loss_ablation_group.py`
- **Automation**: `scripts/run_loss_ablation.sh`

## Support

If you encounter issues:
1. Check the configuration file for syntax errors
2. Verify all dependencies are installed
3. Review logs in the group output directory
4. Create an issue on GitHub with error details
