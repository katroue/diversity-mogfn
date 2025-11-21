# Baseline Comparison Scripts

Scripts for running baseline algorithm comparisons against MOGFN-PC.

## Available Baselines

1. **Random Sampling** - Random trajectory sampling without learning
2. **NSGA-II** - Classic multi-objective genetic algorithm (requires `pymoo`)
3. **HN-GFN** - Hypernetwork-GFlowNet (NeurIPS 2023)
4. **MOGFN-PC** - Multi-Objective GFlowNet with Preference Conditioning (ICML 2023)

## Quick Start

### Run HN-GFN on a Specific Task

The following commands will add HN-GFN results to `results/baselines/{task}/`:

#### HyperGrid (4,000 iterations - default)
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/hypergrid
```

**Output:**
```
results/baselines/hypergrid/
├── hngfn_seed42/
│   ├── checkpoint.pt
│   ├── metrics.json
│   ├── objectives.npy
│   ├── pareto_front.npy
│   └── training_history.json
├── hngfn_seed153/
├── ... (other seeds)
├── hngfn_results.csv
├── all_results.csv
└── summary_by_algorithm.csv
```

#### N-grams (8,000 iterations - default)
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task ngrams \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/ngrams
```

#### Molecules (10,000 iterations - default)
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task molecules \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/molecules
```

#### Sequences (20,000 iterations - default)
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task sequences \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/sequences
```

### Custom Architecture

Control the HN-GFN architecture with these parameters:

```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --hidden_dim 128 \
    --num_layers 4 \
    --z_hidden_dim 64 \
    --z_num_layers 3 \
    --output_dir results/baselines/hypergrid_large
```

**Parameters:**
- `--hidden_dim`: Hidden dimension for policy networks (default: 128)
- `--num_layers`: Number of layers for policy networks (default: 4)
- `--z_hidden_dim`: Hidden dimension for Z hypernetwork (default: 64)
- `--z_num_layers`: Number of layers for Z hypernetwork (default: 3)
- `--eval_samples`: Number of evaluation samples (default: 1000)

### Compare Multiple Algorithms

Run all baselines together:

```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms random,nsga2,mogfn_pc,hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/hypergrid_all
```

Compare MOGFN-PC vs HN-GFN:

```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms mogfn_pc,hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/mogfn_vs_hngfn
```

## Task-Specific Defaults

The script uses validated defaults from factorial experiments:

| Task      | Iterations | Batch Size |
|-----------|-----------|------------|
| hypergrid | 4,000     | 128        |
| ngrams    | 8,000     | 128        |
| molecules | 10,000    | 128        |
| sequences | 20,000    | 128        |

Override with `--num_iterations` and `--batch_size`:

```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42 \
    --num_iterations 1000 \
    --batch_size 32 \
    --output_dir results/baselines/test
```

## Output Files

Each experiment creates:

### Per-Seed Directory (`{algorithm}_seed{seed}/`)
- `checkpoint.pt` - Model weights (MOGFN-PC, HN-GFN only)
- `metrics.json` - All computed metrics
- `objectives.npy` - Sampled objective values
- `pareto_front.npy` - Pareto optimal solutions (NSGA-II, HN-GFN only)
- `training_history.json` - Training progress

### Summary Files
- `{algorithm}_results.csv` - Results for one algorithm across seeds
- `all_results.csv` - Combined results for all algorithms
- `summary_by_algorithm.csv` - Statistical summary (mean, std) by algorithm

## Computed Metrics

All baselines compute applicable metrics:

### Traditional Metrics (all baselines)
- Hypervolume
- Spacing
- R2 Indicator
- Average Pairwise Distance
- Spread

### Spatial Metrics (all baselines)
- Mode Coverage Entropy (MCE)
- Pairwise Minimum Distance (PMD)
- Pareto Front Smoothness (PFS)
- Number of Unique Solutions

### Objective Metrics (all baselines)
- Preference-Aligned Spread (PAS)

### Trajectory Metrics (MOGFN-PC, HN-GFN only)
- Trajectory Diversity Score (TDS)
- Multi-Path Diversity (MPD)

### Dynamics Metrics (MOGFN-PC, HN-GFN only)
- Replay Buffer Diversity (RBD)

### Flow Metrics (MOGFN-PC, HN-GFN only)
- Flow Concentration Index (FCI)

### Composite Metrics (all baselines)
- Quality-Diversity Score (QDS)
- Diversity-Efficiency Ratio (DER)

## Examples by Use Case

### 1. Quick Test (Single Seed, Few Iterations)
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42 \
    --num_iterations 500 \
    --batch_size 16 \
    --output_dir results/baselines/test
```

### 2. Full Experiment (5 Seeds, Default Iterations)
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/hypergrid
```

### 3. Architecture Study (Vary Z Network Size)
```bash
# Small Z network
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --z_hidden_dim 32 \
    --z_num_layers 2 \
    --output_dir results/baselines/hngfn_small_z

# Large Z network
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42,153,264,375,486 \
    --z_hidden_dim 128 \
    --z_num_layers 4 \
    --output_dir results/baselines/hngfn_large_z
```

### 4. All Tasks Comparison
```bash
for task in hypergrid ngrams molecules sequences; do
    python scripts/baselines/run_baseline_comparison.py \
        --task $task \
        --algorithms hngfn \
        --seeds 42,153,264,375,486 \
        --output_dir results/baselines/$task
done
```

### 5. Algorithm Comparison
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms random,nsga2,mogfn_pc,hngfn \
    --seeds 42,153,264,375,486 \
    --output_dir results/baselines/comparison
```

## Viewing Results

### Check Metrics for a Specific Run
```bash
cat results/baselines/hypergrid/hngfn_seed42/metrics.json
```

### View Summary Statistics
```bash
cat results/baselines/hypergrid/summary_by_algorithm.csv
```

### Load Results in Python
```python
import pandas as pd
import numpy as np

# Load all results
df = pd.read_csv('results/baselines/hypergrid/all_results.csv')

# Load objectives for a specific seed
objectives = np.load('results/baselines/hypergrid/hngfn_seed42/objectives.npy')

# Load Pareto front
pareto = np.load('results/baselines/hypergrid/hngfn_seed42/pareto_front.npy')

# Load HN-GFN checkpoint
import torch
checkpoint = torch.load('results/baselines/hypergrid/hngfn_seed42/checkpoint.pt')
```

## Troubleshooting

### NSGA-II Not Working
NSGA-II requires the `pymoo` library:
```bash
pip install pymoo
```

### Out of Memory
Reduce batch size or evaluation samples:
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42 \
    --batch_size 32 \
    --eval_samples 500 \
    --output_dir results/baselines/test
```

### Slow Training
Reduce iterations or use smaller architecture:
```bash
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms hngfn \
    --seeds 42 \
    --num_iterations 1000 \
    --hidden_dim 64 \
    --num_layers 3 \
    --z_hidden_dim 32 \
    --z_num_layers 2 \
    --output_dir results/baselines/test
```

## Help

View all available options:
```bash
python scripts/baselines/run_baseline_comparison.py --help
```

## References

- **HN-GFN**: Zhu et al. "Sample-efficient Multi-objective Molecular Optimization with GFlowNets" (NeurIPS 2023)
- **MOGFN-PC**: Jain et al. "Multi-Objective GFlowNets" (ICML 2023)
- **NSGA-II**: Deb et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II" (IEEE TEC 2002)
