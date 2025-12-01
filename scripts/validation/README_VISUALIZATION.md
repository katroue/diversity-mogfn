# HyperGrid Mode Visualization

## Overview

The `visualize_hypergrid_modes.py` script visualizes discovered modes on the HyperGrid environment from experimental results (ablation studies or factorial experiments).

## What It Does

For HyperGrid experiments, the script:

1. **Loads experimental results**: Reads `objectives.npy`, `preferences.npy`, and `metrics.json` from experiment directories
2. **Infers grid positions**: Approximates (x,y) positions on the grid from objective values
3. **Generates visualizations**:
   - **Objective space scatter plot**: Shows the distribution of solutions in 2D objective space
   - **Grid heatmap**: Displays mode density on the actual HyperGrid with corner markers
   - **Comparison grids**: Side-by-side comparisons of multiple experiments
   - **Reward landscape**: Optional visualization of the reward functions

## Usage

### Single Experiment

Visualize one experiment in detail:

```bash
python scripts/validation/visualize_hypergrid_modes.py \
    --experiment results/ablations/capacity/small_concat_seed42 \
    --output_dir results/validation/mode_visualizations
```

### Compare Multiple Experiments

Compare specific experiments side-by-side:

```bash
python scripts/validation/visualize_hypergrid_modes.py \
    --experiments results/ablations/capacity/small_concat_seed42 \
                 results/ablations/capacity/medium_concat_seed42 \
                 results/ablations/capacity/large_concat_seed42 \
                 results/ablations/capacity/xlarge_concat_seed42 \
    --output_dir results/validation/capacity_comparison
```

### Visualize All Experiments from Ablation Study

Process all experiments in a directory:

```bash
python scripts/validation/visualize_hypergrid_modes.py \
    --ablation_dir results/ablations/capacity \
    --output_dir results/validation/capacity_modes
```

**Note**: For large ablation studies (50+ experiments), this may take several minutes.

### Show Reward Landscape

Add `--show_landscape` to visualize the underlying reward functions:

```bash
python scripts/validation/visualize_hypergrid_modes.py \
    --experiment results/ablations/capacity/small_concat_seed42 \
    --show_landscape \
    --output_dir results/validation/mode_visualizations
```

### Factorial Experiments

Works identically with factorial results:

```bash
python scripts/validation/visualize_hypergrid_modes.py \
    --ablation_dir results/factorials/hypergrid \
    --output_dir results/validation/factorial_modes
```

### Baseline Comparisons

Visualize and compare baseline algorithms (MOGFN-PC, HN-GFN, NSGA-II, Random):

```bash
# All baselines in directory
python scripts/validation/visualize_hypergrid_modes.py \
    --ablation_dir results/baselines/hypergrid \
    --output_dir results/validation/baseline_modes

# Compare specific baselines
python scripts/validation/visualize_hypergrid_modes.py \
    --experiments results/baselines/hypergrid/mogfn_pc_seed42 \
                 results/baselines/hypergrid/hngfn_seed42 \
                 results/baselines/hypergrid/nsga2_seed42 \
                 results/baselines/hypergrid/random_seed42 \
    --output_dir results/validation/baseline_comparison
```

**Note**: Baseline methods (NSGA-II, Random) don't have `preferences.npy` files - the script handles this automatically.

### Aggregating Seeds

By default, the script creates one visualization per experiment directory. For baselines with multiple seeds (5 seeds × 4 algorithms = 20 experiments), use `--aggregate_seeds` to combine all seeds for each algorithm:

```bash
# Creates only 4 plots (one per algorithm) instead of 20 (one per seed)
python scripts/validation/visualize_hypergrid_modes.py \
    --ablation_dir results/baselines/hypergrid \
    --aggregate_seeds \
    --output_dir results/validation/baseline_modes_aggregated
```

This combines objectives from all 5 seeds for each algorithm, giving a more comprehensive view of mode discovery patterns.

**Outputs**:
- `{algorithm}_aggregated_modes.pdf` - Individual visualizations (e.g., `hngfn_aggregated_modes.pdf`)
- `mode_comparison_aggregated.pdf` - Side-by-side comparison of all algorithms

## Output Files

For each experiment, the script generates:

- **`{exp_name}_modes.pdf`**: Detailed visualization with objective space + grid heatmap
- **`{exp_name}_modes.png`**: PNG version of the same visualization
- **`mode_comparison.pdf`**: Comparison grid (when visualizing multiple experiments)
- **`reward_landscape.pdf`**: Reward function visualization (if `--show_landscape` used)

## How Position Inference Works

For the standard 2-objective HyperGrid with "corners" configuration:

- **Objective 1**: Reward function peaked at top-right corner (7,7)
- **Objective 2**: Reward function peaked at top-left corner (0,7)

The reward functions are:
```
R1(x,y) = exp(-0.5 * dist_to_(7,7))
R2(x,y) = exp(-0.5 * dist_to_(0,7))
```

The script infers approximate (x,y) positions by:
1. Using objective values as weights towards their respective corners
2. Solving inverse distance equations
3. Clamping to grid boundaries [0, 7]

**Note**: Position inference is approximate. The visualization shows general mode distribution patterns rather than exact grid positions.

## Interpreting Visualizations

### Objective Space Plot (Left Panel)

- **Scatter points**: Each dot = one sampled solution
- **Axes**: Objective 1 (top-right reward) vs Objective 2 (top-left reward)
- **Patterns**:
  - Points near (1, 0): Solutions at top-right corner
  - Points near (0, 1): Solutions at top-left corner
  - Points on diagonal: Solutions between corners (Pareto front)
  - Clustered points: Repeated discovery of same mode

### Grid Heatmap (Right Panel)

- **Color intensity**: Number of samples discovered at each grid cell
- **Red star (★)**: Top-right corner (Objective 1 peak)
- **Green star (★)**: Top-left corner (Objective 2 peak)
- **Patterns**:
  - Concentrated at corners: Model found high-reward regions
  - Uniform distribution: Model explores broadly
  - Sparse areas: Under-explored regions

### Comparison Grid

When comparing experiments:
- Each subplot shows one experiment's mode distribution
- Titles include: experiment name, capacity, conditioning type
- Compare heat patterns to see how hyperparameters affect exploration

## Examples

### Example 1: Capacity Ablation

```bash
# Compare how model capacity affects mode discovery
python scripts/validation/visualize_hypergrid_modes.py \
    --experiments results/ablations/capacity/small_concat_seed42 \
                 results/ablations/capacity/medium_concat_seed42 \
                 results/ablations/capacity/large_concat_seed42 \
                 results/ablations/capacity/xlarge_concat_seed42 \
    --output_dir results/validation/capacity_comparison
```

**Expected pattern**: Larger models may show more diverse mode coverage.

### Example 2: Temperature Comparison

```bash
# Compare sampling temperature effects
python scripts/validation/visualize_hypergrid_modes.py \
    --experiments results/ablations/sampling/temp_low_seed42 \
                 results/ablations/sampling/temp_high_seed42 \
                 results/ablations/sampling/temp_very_high_seed42 \
    --output_dir results/validation/temperature_comparison
```

**Expected pattern**: Higher temperature may lead to more exploration (broader distribution).

### Example 3: Conditioning Methods

```bash
# Compare concat vs FiLM conditioning
python scripts/validation/visualize_hypergrid_modes.py \
    --experiments results/ablations/capacity/medium_concat_seed42 \
                 results/ablations/capacity/medium_film_seed42 \
    --output_dir results/validation/conditioning_comparison
```

**Expected pattern**: Different conditioning may affect preference-conditioned sampling.

### Example 4: Baseline Algorithm Comparison

```bash
# Compare all baseline algorithms
python scripts/validation/visualize_hypergrid_modes.py \
    --experiments results/baselines/hypergrid/mogfn_pc_seed42 \
                 results/baselines/hypergrid/hngfn_seed42 \
                 results/baselines/hypergrid/nsga2_seed42 \
                 results/baselines/hypergrid/random_seed42 \
    --output_dir results/validation/baseline_comparison
```

**Expected patterns**:
- **Random**: Uniform distribution across grid (poor mode coverage)
- **NSGA-II**: Concentration at corners (Pareto front discovery)
- **HN-GFN**: Similar to MOGFN-PC but may differ in diversity
- **MOGFN-PC**: Preference-conditioned exploration with diverse modes

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--experiment` | Path | - | Single experiment directory |
| `--experiments` | Path(s) | - | Multiple experiment directories |
| `--ablation_dir` | Path | - | Directory containing all experiments |
| `--output_dir` | Path | `results/validation/mode_visualizations` | Output directory |
| `--grid_height` | int | 8 | Grid size (H×H) |
| `--num_objectives` | int | 2 | Number of objectives |
| `--reward_config` | str | `corners` | Reward config (`corners` or `modes`) |
| `--show_landscape` | flag | False | Generate reward landscape plot |
| `--aggregate_seeds` | flag | False | Aggregate all seeds for each algorithm/config |

## Limitations

1. **2-objective focus**: Grid position inference only works for 2-objective HyperGrid
   - For 3+ objectives, script shows parallel coordinates plot instead
2. **Approximate positions**: Inferred grid positions are estimates, not exact
3. **Corners config only**: Assumes standard "corners" reward configuration
4. **Large datasets**: Processing 50+ experiments may be slow
   - Baseline experiments often have 100K+ samples (slower visualization)
   - Consider using `--experiments` with specific seeds instead of `--ablation_dir`

## Integration with Other Scripts

This visualization complements:

- **`train_predictive_models.py`**: Visualize modes for high/low QDS experiments
- **`compute_metric_correlations.py`**: Visualize experiments with high/low diversity metrics
- **`metric_factor_analysis.py`**: Visual validation of factor analysis results

## Requirements

The script requires:
- `numpy` - Load objectives/preferences arrays
- `matplotlib` - Generate visualizations
- `seaborn` - Enhanced plot styling
- `scipy` - Distance computations (via metrics modules)

All dependencies are already in the project's virtual environment.

## Troubleshooting

### Error: "objectives.npy not found"
- Ensure experiment has completed and saved results
- Check experiment directory structure

### Error: "metrics.json not found"
- Older experiments may lack metadata
- Re-run experiment with latest code

### Blank/empty visualizations
- Check that objectives array is not all zeros
- Verify experiment converged (check `training_history.json`)

### Position inference looks wrong
- Verify `--grid_height` matches experiment config
- Ensure using "corners" reward configuration
- For custom configs, position inference may be inaccurate

## Future Enhancements

Potential additions:
- [ ] Support for "modes" reward configuration
- [ ] Exact state extraction from checkpoint
- [ ] Trajectory path visualization
- [ ] Animation of mode discovery over training
- [ ] Preference-conditioned mode grouping
