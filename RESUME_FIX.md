# Resume Functionality Fix

## Problem

The `--resume` flag in factorial experiment scripts was **losing previously completed experiments** when resuming interrupted runs.

### Root Cause

In both factorial experiment runners:
- `scripts/factorials/hypergrid/run_factorial_experiment.py`
- `scripts/factorials/ngrams/run_factorial_experiment_ngrams.py`

The incremental save logic (lines ~539-540) was:

```python
# Save incremental results
df_temp = pd.DataFrame(results)  # Only contains NEW results from current run!
df_temp.to_csv(results_temp_file, index=False)  # OVERWRITES old file!
```

### Execution Flow Causing Data Loss

1. **First run** (5 experiments):
   - Results list: `[exp1, exp2, exp3, exp4, exp5]`
   - Saves to `results_temp.csv`: ✓ 5 experiments

2. **Resume second run** (3 more experiments):
   - Loads completed from CSV: `{exp1, exp2, exp3, exp4, exp5}`
   - **Results list starts empty**: `[]`
   - Skips completed experiments ✓
   - Runs new experiments: `results = [exp6, exp7, exp8]`
   - **Saves to CSV**: `[exp6, exp7, exp8]` ← OVERWRITES!
   - **Lost**: `exp1-exp5` ❌

3. **Resume third run**:
   - Only finds 3 completed (exp6, exp7, exp8)
   - Re-runs exp1-exp5 unnecessarily

### Impact

- Wasted computation: experiments run multiple times
- Confusing progress: completion count resets
- Incomplete data: final results missing earlier experiments

## Solution

Merge with existing `results_temp.csv` before saving, similar to how the final `results.csv` is handled:

```python
# Save incremental results
df_temp = pd.DataFrame(results)

# Merge with existing temp results if they exist
if results_temp_file.exists():
    try:
        df_existing = pd.read_csv(results_temp_file)
        df_temp = pd.concat([df_existing, df_temp], ignore_index=True)
        # Remove duplicates (keep last occurrence in case of re-runs)
        df_temp = df_temp.drop_duplicates(subset=['exp_name'], keep='last')
    except Exception as e:
        # If reading fails, just save the new results
        tqdm.write(f"  ⚠ Warning: Could not merge with existing temp results: {e}")

df_temp.to_csv(results_temp_file, index=False)
```

### Key Improvements

1. **Preserves old results**: Reads existing CSV and merges with new results
2. **Handles re-runs**: `drop_duplicates(keep='last')` updates experiment if re-run
3. **Error handling**: Falls back to saving new results if merge fails
4. **Consistent with final save**: Uses same merge logic as final `results.csv` (lines ~558-560)

## Verification

Run the test:

```bash
python tests/test_resume_fix.py
```

Expected output:
```
✓ ALL TESTS PASSED - Resume functionality fix works correctly!

The fix ensures that:
  • Previously completed experiments are preserved
  • New experiments are appended to results_temp.csv
  • Re-running an experiment updates its values (keeps last)
  • No data loss when resuming interrupted experiments
```

## Usage

Now `--resume` works correctly:

```bash
# HyperGrid factorial (capacity × loss)
python scripts/factorials/hypergrid/run_factorial_experiment.py \
    --config configs/factorials/capacity_loss_2way.yaml \
    --output_dir results/factorials/capacity_loss \
    --resume

# NGrams factorial (capacity × loss)
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_capacity_loss_2way.yaml \
    --resume
```

The script will:
- Load all previously completed experiments from `results_temp.csv`
- Skip those experiments
- Run only the remaining experiments
- **Preserve all results** (old + new) in `results_temp.csv`

## Testing the Fix

To verify the fix is working in your actual run:

1. Run with `--dry-run` first:
   ```bash
   python scripts/factorials/hypergrid/run_factorial_experiment.py \
       --config configs/factorials/capacity_loss_2way.yaml \
       --output_dir results/factorials/capacity_loss \
       --resume --dry-run
   ```

2. Check the output:
   ```
   Resuming from temporary results: results/factorials/capacity_loss/results_temp.csv
   Found 15 completed experiments  ← Should show all previously completed

   [SKIP - COMPLETED] small_tb_seed42
   [SKIP - COMPLETED] small_tb_seed153
   ...
   Total to skip: 15  ← Should match actual completed count
   ```

3. Run without `--dry-run` when verified

## Files Modified

- `scripts/factorials/hypergrid/run_factorial_experiment.py` (line ~539-540)
- `scripts/factorials/ngrams/run_factorial_experiment_ngrams.py` (line ~560-574)
- `tests/test_resume_fix.py` (new test)
