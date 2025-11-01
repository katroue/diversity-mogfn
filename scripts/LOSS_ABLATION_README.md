# Loss Ablation Study - Group-by-Group Execution

This guide explains how to run the loss ablation study one experiment group at a time for better analysis and resource management.

## Overview

The loss ablation study is organized into **5 experiment groups**, each testing different aspects of loss functions:

1. **base_loss_comparison** (30 runs) - Compare core GFlowNet losses
2. **entropy_regularization** (25 runs) - Test entropy regularization strengths
3. **kl_regularization** (15 runs) - Test KL divergence regularization
4. **subtb_entropy_sweep** (20 runs) - SubTB + entropy combinations
5. **loss_modifications** (15 runs) - Test loss computation modifications

**Total: 105 experiments** (21 configs × 5 seeds)

## Quick Start

### List All Available Groups

```bash
python scripts/run_loss_ablation_group.py --list
```

This shows all groups, their descriptions, and the number of experiments in each.

### Run a Specific Group (Recommended)

```bash
# Run the first group
python scripts/run_loss_ablation_group.py --group base_loss_comparison

# Preview what would run (dry-run)
python scripts/run_loss_ablation_group.py --group base_loss_comparison --dry-run

# Resume a group that was interrupted
python scripts/run_loss_ablation_group.py --group base_loss_comparison --resume
```

### Run All Groups Sequentially

```bash
# Run all 5 groups one after another
python scripts/run_loss_ablation_group.py --all

# With resume (skip completed experiments)
python scripts/run_loss_ablation_group.py --all --resume
```

## Recommended Execution Order

Run groups in this order, as results from earlier groups inform later ones:

### Week 1: Base Loss Comparison

```bash
python scripts/run_loss_ablation_group.py --group base_loss_comparison
```

**What it tests**: 6 different base GFlowNet losses
- Trajectory Balance (TB)
- Detailed Balance (DB)
- SubTrajectory Balance with λ=0.5, 0.9, 0.95
- Flow Matching (FM)

**Why run first**: Identifies the best base loss to use in subsequent groups.

**Expected time**: ~2 hours with 10 parallel jobs

**After completion**: Analyze results to determine best base loss before proceeding.

### Week 2: Entropy Regularization

```bash
python scripts/run_loss_ablation_group.py --group entropy_regularization
```

**What it tests**: 5 entropy regularization strengths (β = 0, 0.01, 0.05, 0.1, 0.5)

**Why run second**: Tests how entropy affects diversity on the best base loss.

**Expected time**: ~1.5 hours with 10 parallel jobs

**After completion**: Identify optimal β value.

### Week 3: KL Regularization

```bash
python scripts/run_loss_ablation_group.py --group kl_regularization
```

**What it tests**: 3 KL divergence strengths (β = 0, 0.01, 0.1)

**Why run third**: Alternative regularization approach to entropy.

**Expected time**: ~45 minutes with 10 parallel jobs

### Week 4: SubTB + Entropy Combinations

```bash
python scripts/run_loss_ablation_group.py --group subtb_entropy_sweep
```

**What it tests**: SubTB(λ=0.9) with 4 entropy levels

**Why run fourth**: Hypothesis testing - is SubTB+Entropy the best combination?

**Expected time**: ~1 hour with 10 parallel jobs

### Week 5: Loss Modifications

```bash
python scripts/run_loss_ablation_group.py --group loss_modifications
```

**What it tests**: 3 loss computation modifications
- Standard
- Temperature scaling
- Reward shaping with diversity bonus

**Why run last**: Tests advanced modifications on best loss+regularization combo.

**Expected time**: ~45 minutes with 10 parallel jobs

## Output Structure

Results are organized by group:

```
results/ablations/loss/
├── base_loss_comparison/
│   ├── results.csv                      # All metrics for this group
│   ├── group_config.yaml                # Group configuration
│   ├── trajectory_balance_seed42_config.json
│   ├── trajectory_balance_seed42/       # Individual experiment outputs
│   ├── detailed_balance_seed42_config.json
│   └── ...
├── entropy_regularization/
│   ├── results.csv
│   └── ...
└── ...
```

## Analyzing Results Between Groups

After each group completes, analyze results before proceeding:

```bash
# Load and examine results
cd results/ablations/loss/<group_name>/

# View summary statistics
python -c "
import pandas as pd
df = pd.read_csv('results.csv')
print(df.groupby('name')[['hypervolume', 'tds', 'mce', 'pas']].mean())
"
```

## Resume from Interruptions

If a group run is interrupted:

```bash
python scripts/run_loss_ablation_group.py --group <group_name> --resume
```

This will:
- Read existing `results.csv`
- Skip experiments already completed
- Continue with remaining experiments

## Advanced Options

### Custom Output Directory

```bash
python scripts/run_loss_ablation_group.py \
    --group base_loss_comparison \
    --output_dir /path/to/custom/output
```

### Dry Run (Preview)

```bash
python scripts/run_loss_ablation_group.py \
    --group base_loss_comparison \
    --dry-run
```

Shows what experiments would be run without actually executing them.

## Integration with Main Ablation Script

This script generates configurations compatible with `run_ablation_study.py`:

```bash
# Alternative: Use main ablation script for entire study
python scripts/run_ablation_study.py \
    --config configs/ablations/loss_ablation.yaml \
    --ablation loss \
    --output_dir results/ablations/loss
```

However, the group-by-group approach (`run_loss_ablation_group.py`) is recommended for:
- **Better resource management**: Run smaller batches
- **Interim analysis**: Analyze results between groups
- **Adaptive execution**: Adjust later groups based on earlier results
- **Easier debugging**: Isolate issues to specific groups

## Expected Outcomes

Based on hypotheses in `loss_ablation.yaml`:

1. **Base Loss**: SubTB(λ=0.9) expected to outperform TB, DB, FM
2. **Entropy**: β=0.05 expected to increase diversity by 20-30%
3. **Too much regularization**: β=0.5 expected to hurt quality
4. **Best combination**: SubTB(0.9) + Entropy(β=0.05)

## Troubleshooting

### Group Fails to Start

Check that configuration file exists:
```bash
ls configs/ablations/loss_ablation.yaml
```

### Individual Experiments Failing

Check `failed.json` in group directory:
```bash
cat results/ablations/loss/<group_name>/failed.json
```

### Unexpected Results

1. Verify configurations: Check `<group_name>/group_config.yaml`
2. Examine individual outputs: Check experiment directories
3. Compare with baselines: Review previous ablation results

## Next Steps

After completing all groups:

1. **Aggregate results**: Combine all group results
2. **Statistical analysis**: Run ANOVA, post-hoc tests
3. **Visualizations**: Create comparison plots
4. **Select best configuration**: Use for factorial experiments
5. **Update documentation**: Record findings in paper draft

## Citation

If you use this ablation study structure in your work, please cite:

```
@article{diversity-mogfn-2025,
  title={Diversity Mechanisms in Multi-Objective GFlowNets},
  author={...},
  year={2025}
}
```
