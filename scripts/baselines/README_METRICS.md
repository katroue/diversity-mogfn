# Adding Missing Metrics to Baseline Results

## Overview

The `add_missing_metrics.py` script computes missing convergence metrics (GD, IGD) for baseline experiment results and updates all CSV files.

## Missing Metrics

The following metrics are computed and added:

1. **GD (Generational Distance)**: Measures average distance from obtained solutions to reference Pareto front
   - Lower is better (closer to reference front)
   - Requires reference Pareto front

2. **IGD (Inverted Generational Distance)**: Measures average distance from reference Pareto front to obtained solutions
   - Lower is better (better coverage of reference front)
   - Requires reference Pareto front

## How It Works

1. **Scans experiment directories** in `results/baselines/*/`
2. **Loads objectives** from all experiments (`objectives.npy`)
3. **Computes empirical Pareto front** from all algorithms combined
4. **Calculates GD/IGD** for each experiment using the empirical front as reference
5. **Updates individual metrics.json** files with new metrics
6. **Regenerates CSVs**:
   - `all_results.csv` - All experiments with all metrics
   - `summary_by_algorithm.csv` - Mean/std by algorithm

## Usage

### Process single task directory:
```bash
python scripts/baselines/add_missing_metrics.py --results_dir results/baselines/hypergrid
```

### Process all task directories recursively:
```bash
python scripts/baselines/add_missing_metrics.py --results_dir results/baselines --recursive
```

## Output

The script will:
- Add `gd` and `igd` fields to each `metrics.json`
- Regenerate `all_results.csv` with gd/igd columns
- Regenerate `summary_by_algorithm.csv` with gd/igd statistics

### Example Output:

```
======================================================================
Processing: results/baselines/hypergrid
======================================================================
Found 15 experiment directories
Computing empirical Pareto front from all experiments...
  Empirical Pareto front: 8 solutions from 645500 total

Adding GD/IGD metrics to experiments...
    mogfn_pc_seed42: GD=0.108452, IGD=0.139113
    nsga2_seed42: GD=0.000000, IGD=0.000000
    random_seed42: GD=0.211081, IGD=0.000000

Updated 15 experiments with GD/IGD metrics

Regenerating all_results.csv...
Regenerated results/baselines/hypergrid/all_results.csv
Regenerating summary_by_algorithm.csv...
Regenerated results/baselines/hypergrid/summary_by_algorithm.csv

GD/IGD Summary by Algorithm:
                 gd                       igd
               mean       std count      mean       std count
algorithm
mogfn_pc   0.110719  0.003825     5  0.083468  0.076195     5
nsga2      0.000000  0.000000     5  0.000000  0.000000     5
random     0.211079  0.000075     5  0.000000  0.000000     5
```

## Interpretation

- **NSGA-II**: GD=0, IGD=0 means it found solutions on the empirical Pareto front (expected for Pareto-optimizing algorithm)
- **Random**: High GD (0.21) means solutions far from Pareto front
- **MOGFN-PC**: Intermediate GD (0.11) shows reasonable convergence

## Notes

- The script uses an **empirical Pareto front** computed from all algorithms combined
- If true Pareto front is known for a task, it could be substituted
- The script is **idempotent** - running multiple times won't duplicate metrics
- Already-computed GD/IGD values are preserved (not recomputed)

## Files Modified

For each task directory (e.g., `results/baselines/hypergrid/`):
- `{algorithm}_seed{seed}/metrics.json` - Individual experiment metrics
- `all_results.csv` - Complete results table
- `summary_by_algorithm.csv` - Aggregated statistics

## Requirements

All metrics modules from `src/metrics/`:
- `traditional.py` - For GD/IGD computation
