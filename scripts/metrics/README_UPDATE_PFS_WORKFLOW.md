# Workflow: Updating PFS Metrics in All Results

This guide explains how to update PFS (Pareto Front Smoothness) values across all your experiment results using the extended implementation.

## Overview

The PFS metric has been extended to support 3+ objectives via manifold projection. To update existing results, follow this two-step process:

1. **Recalculate PFS** in individual experiment `metrics.json` files
2. **Update CSV files** with the new PFS values

## Step 1: Recalculate PFS in metrics.json Files

The `recalculate_pfs.py` script updates individual experiment directories:

```bash
# Preview changes (dry run)
python scripts/metrics/recalculate_pfs.py \
    --results_dir results/ablations/capacity \
    --dry_run

# Actually update experiments
python scripts/metrics/recalculate_pfs.py \
    --results_dir results/ablations/capacity
```

### What This Does

- Finds all experiments with `objectives.npy` files
- Recalculates PFS using the extended implementation
- Updates each experiment's `metrics.json` file
- Creates `metrics_backup.json` before making changes
- Adds metadata about the recalculation

### Example Output

```
Recalculating PFS...
[1/40] ✓ small_concat_seed42
           Old PFS: 0.000000
           New PFS: 0.015274
           Diff: 0.015274 ✓ SAVED

[2/40] ✓ small_concat_seed153
           Old PFS: 0.000000
           New PFS: 0.018562
           Diff: 0.018562 ✓ SAVED
...

SUMMARY
Total experiments:  40
Updated:            38
Unchanged:          2
Errors:             0
```

## Step 2: Update CSV Results Files

The `update_pfs_in_csvs.py` script propagates changes to aggregated CSV files:

```bash
# Preview changes (dry run)
python scripts/metrics/update_pfs_in_csvs.py \
    --results_dir results/ablations/capacity \
    --dry_run

# Actually update CSV files
python scripts/metrics/update_pfs_in_csvs.py \
    --results_dir results/ablations/capacity
```

### What This Does

- Loads updated PFS values from all `metrics.json` files
- Finds all CSV files in the results directory:
  - `all_results.csv`
  - `results.csv`
  - `results_temp.csv`
  - `summary*.csv`
- Updates PFS column in each CSV
- Creates timestamped backups before saving

### Example Output

```
Loading PFS values from experiment metrics.json files...
Found 40 experiments with PFS values

Found 1 CSV files to update

Processing: all_results.csv
  ✓ Updated 38/40 rows
     Old PFS mean: 0.000072
     New PFS mean: 0.051811
     ✓ SAVED

SUMMARY
Total CSV files:      1
Updated:              1
Unchanged:            0
```

## Complete Workflow for All Ablations

To update all ablation studies:

```bash
# 1. Capacity ablation
python scripts/metrics/recalculate_pfs.py --results_dir results/ablations/capacity
python scripts/metrics/update_pfs_in_csvs.py --results_dir results/ablations/capacity

# 2. Sampling ablation
python scripts/metrics/recalculate_pfs.py --results_dir results/ablations/sampling
python scripts/metrics/update_pfs_in_csvs.py --results_dir results/ablations/sampling

# 3. Loss ablation
python scripts/metrics/recalculate_pfs.py --results_dir results/ablations/loss
python scripts/metrics/update_pfs_in_csvs.py --results_dir results/ablations/loss

# 4. Baselines
python scripts/metrics/recalculate_pfs.py --results_dir results/baselines
python scripts/metrics/update_pfs_in_csvs.py --results_dir results/baselines
```

Or use a loop:

```bash
for dir in results/ablations/*/; do
    python scripts/metrics/recalculate_pfs.py --results_dir "$dir"
    python scripts/metrics/update_pfs_in_csvs.py --results_dir "$dir"
done
```

## Safety Features

### Backups

Both scripts create backups before making changes:

- **recalculate_pfs.py**: Creates `metrics_backup.json` (one-time, won't overwrite existing backup)
- **update_pfs_in_csvs.py**: Creates timestamped CSV backups (e.g., `all_results_backup_20251201_143522.csv`)

### Dry Run Mode

Always test with `--dry_run` first to preview changes:

```bash
# See what would change without saving
python scripts/metrics/recalculate_pfs.py --results_dir results/ablations/capacity --dry_run
python scripts/metrics/update_pfs_in_csvs.py --results_dir results/ablations/capacity --dry_run
```

### Metadata Tracking

Updated `metrics.json` files include metadata:

```json
{
  "pfs": 0.015274,
  "pfs_metadata": {
    "recalculated": true,
    "recalculation_date": "2025-12-01T20:46:21",
    "old_value": 0.000000,
    "method": "extended_pfs_with_manifold_projection"
  }
}
```

## Expected Changes

### For 2-Objective Problems (HyperGrid)

Most experiments will see PFS change from 0.0 to small positive values (0.001-0.03):

- **Before**: Many edge cases returned 0.0
- **After**: Extended implementation handles edge cases better
- **Interpretation**: Lower PFS still = smoother front

### For 3-Objective Problems (DNA Sequences)

All experiments will see changes:

- **Before**: Only measured first 2 objectives (2D slice)
- **After**: Measures full 3D Pareto surface via 2D projection
- **Impact**: Captures complete manifold structure

## Verification

After updating, verify the changes:

```bash
# Check a specific experiment
cat results/ablations/capacity/small_concat_seed42/metrics.json | grep -A 6 pfs

# Compare old and new CSV
diff results/ablations/capacity/all_results.csv \
     results/ablations/capacity/all_results_backup_*.csv
```

## Troubleshooting

### "No experiments found"

- Check that the results directory exists
- Ensure experiments have `objectives.npy` and `metrics.json` files

### "No changes needed"

- PFS values are already up to date
- Check if recalculate_pfs.py was already run

### CSV file not found

- Some ablations may not have generated CSV files yet
- Run the ablation's summary script first

## Notes

- The update process is **idempotent** - safe to run multiple times
- Only experiments with changed PFS values are updated
- CSV backups are timestamped to prevent overwriting
- Both scripts work recursively through subdirectories
