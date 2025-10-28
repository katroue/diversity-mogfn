# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A systematic study of diversity in Multi-Objective Generative Flow Networks (MOGFNs), implementing novel GFlowNet-specific diversity metrics and comprehensive ablation studies. Based on "Multi-Objective GFlowNets" by Jain et al. (ICML 2023).

## Development Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# The project uses Python 3.9 with PyTorch
# Dependencies are installed in .venv (already exists)
```

## Running Experiments

### Ablation Studies

The primary workflow involves running ablation studies with different configurations:

```bash
# Capacity ablation (model size: small/medium/large/xlarge)
python scripts/run_ablation_study.py \
    --config configs/ablations/capacity_ablation.yaml \
    --ablation capacity \
    --output_dir results/ablations/capacity

# Sampling ablation (preference sampling strategies)
python scripts/run_ablation_study.py \
    --config configs/ablations/sampling_ablation.yaml \
    --ablation sampling \
    --output_dir results/ablations/sampling

# Loss ablation (different loss functions)
python scripts/run_ablation_study.py \
    --config configs/ablations/loss_ablation_final.yaml \
    --ablation loss \
    --output_dir results/ablations/loss

# Resume from existing results
python scripts/run_ablation_study.py \
    --config configs/ablations/capacity_ablation.yaml \
    --ablation capacity \
    --output_dir results/ablations/capacity \
    --resume
```

### Testing

```bash
# Test capacity ablation (fast, minimal iterations)
python tests/test_ablation_capacity.py

# Test sampling ablation with parameter fixes (validates that different configs produce different results)
# Tests: temperature, sampling strategies, off-policy ratio, preference distribution
# Runs 8 experiments × 500 iterations each (~5-10 minutes)
python tests/test_sampling_ablation.py
```

### Post-Processing

```bash
# Update PAS (Preference-Aligned Spread) metrics for all experiments
python scripts/update_all_pas.py

# Fix and recompute PMD (Pairwise Minimum Distance) metrics
python scripts/fix_and_recompute_pmd.py

# RECOMMENDED: Create comprehensive capacity ablation report
# Generates summary CSVs + 8-page PDF with ALL metrics grouped by capacity & conditioning
python scripts/create_comprehensive_report.py \
    --results_csv results/ablations/capacity/all_results.csv
# Output: results/ablations/reports/
#   - comprehensive_metrics_report.pdf (8 pages)
#   - summary_by_capacity.csv
#   - summary_by_conditioning.csv
#   - summary_by_capacity_and_conditioning.csv
#   - overall_summary.csv
#   - capacity_detailed_results.csv

# Analyze sampling ablation study (interactive Jupyter notebook)
jupyter notebook create_sampling_ablation_report.ipynb
# Analyzes 5 experiment types: temperature, strategies, policy, preference, batch size
# Output: results/ablations/sampling/report/
#   - Summary CSVs for each experiment type
#   - Comparison visualizations (PNGs)
#   - comprehensive_report.txt with recommendations
#   - top_configurations_radar.png

# Alternative: Create simple reports for individual ablation studies
python scripts/create_summary_report.py \
    --results_csv results/ablations/sampling/all_results.csv \
    --ablation sampling \
    --output_dir results/ablations/sampling/report

python scripts/create_metrics_comparison_pdf.py \
    --results_csv results/ablations/sampling/all_results.csv \
    --ablation sampling \
    --output_dir results/ablations/sampling/report
```

## Code Architecture

### Core Components

**Models** (`src/models/`):
- `gflownet.py`: Base GFlowNet with trajectory balance loss, PolicyNetwork, BackwardPolicyNetwork, Trajectory dataclass
- `mogfn_pc.py`: Multi-Objective GFlowNet with Preference Conditioning (MOGFN-PC)
  - `MOGFN_PC`: Main model class
  - `PreferenceEncoder`: Encodes preference vectors (vanilla or thermometer encoding)
  - `PreferenceSampler`: Samples preferences using different distributions (dirichlet/uniform)
  - `MOGFNTrainer`: Training loop with trajectory balance loss
  - `MOGFNSampler`: Sampling trajectories conditioned on preferences
- `conditioning.py`: Preference conditioning mechanisms (concat, FiLM, etc.)

**Environments** (`src/environments/`):
- `hypergrid.py`: H×H grid environment with multi-objective rewards at corners/regions
- `molecules.py`: Molecular generation environment
- `sequences.py`: Sequence generation environment
- `ngrams.py`: N-gram environment

**Metrics** (`src/metrics/`):
The project implements 7 categories of diversity metrics:

1. **Traditional** (`traditional.py`): Standard MO metrics from NSGA-II/MOEA/D
   - Hypervolume, Generational Distance, Inverted Generational Distance, Spacing

2. **Trajectory** (`trajectory.py`): Path-based diversity
   - Trajectory Diversity Score (TDS), Multi-Path Diversity (MPD)

3. **Spatial** (`spatial.py`): Objective space coverage
   - Mode Coverage Entropy (MCE), Pairwise Minimum Distance (PMD)

4. **Objective** (`objective.py`): Preference-aware metrics
   - Preference-Aligned Spread (PAS), Pareto Front Smoothness (PFS)

5. **Dynamics** (`dynamics.py`): Learning dynamics
   - Replay Buffer Diversity (RBD)

6. **Flow** (`flow.py`): Flow-specific metrics
   - Flow Concentration Index (FCI)

7. **Composite** (`composite.py`): Combined quality-diversity
   - Quality-Diversity Score (QDS), Diversity-Efficiency Ratio (DER)

### Configuration Structure

Ablation configs (`configs/ablations/`) define:
- `experiments`: List of experimental variations (e.g., different model capacities, conditioning types)
- `fixed`: Constants shared across all experiments (num_iterations, batch_size, learning_rate, seeds)

Key hyperparameters:
- `capacity`: Model size (small/medium/large/xlarge)
- `hidden_dim`: Hidden layer dimensions (32/64/128/256)
- `num_layers`: Number of network layers (2/3/4)
- `conditioning`: Preference conditioning type (concat/film)
- `preference_sampling`: Distribution for sampling preferences (dirichlet/uniform)
- `alpha`: Dirichlet concentration parameter (typically 1.5)

### Experimental Pipeline

1. **Training**: `run_ablation_study.py` runs experiments for each config × seed combination
   - Creates MOGFN model with specified architecture
   - Trains using trajectory balance loss
   - Evaluates and computes all 7 metric categories
   - Saves: checkpoint.pt, metrics.json, objectives.npy, preferences.npy, training_history.json

2. **Results**: Each experiment saved to `results/ablations/{type}/{exp_name}_seed{seed}/`
   - Intermediate results saved to `results_temp.csv` every 5 experiments
   - Final aggregated results in `all_results.csv`

3. **Analysis**: Summary reports generated with grouping by experimental variables
   - `summary_by_capacity.csv`, `summary_by_conditioning.csv`, etc.
   - Includes mean, std, count for all metrics

### Important Implementation Details

- **Tensor/NumPy handling**: Models operate on PyTorch tensors, metrics on NumPy arrays. Use utilities from `src/utils/tensor_utils.py` for safe conversions:
  - `to_numpy()`: Convert tensors to NumPy arrays
  - `to_tensor()`: Convert NumPy arrays to tensors
  - `to_hashable()`: Convert states to hashable tuples for dict keys
  - `ensure_device()`: Validate tensor device placement
  - `check_nan_inf()`: Validate numerical stability
- **State representation**: HyperGrid uses (x,y) coordinates. Use `to_hashable()` utility to convert states to hashable tuples for tracking
- **PAS computation**: Uses simplified approximation (average pairwise distance) rather than full preference-conditioned sampling for efficiency
- **Path handling**: All scripts use dynamic path resolution via `Path(__file__).parent.parent` for portability

## Tensor/NumPy Conversion Utilities

The project provides utilities in `src/utils/tensor_utils.py` to handle conversions safely:

```python
from src.utils.tensor_utils import to_numpy, to_hashable, validate_same_device

# Convert tensor to numpy for metrics
objectives = to_numpy(objectives_tensor)

# Convert state to hashable key for dicts
state_key = to_hashable(state)

# Validate all tensors on same device
validate_same_device(tensor1, tensor2, tensor3)
```

## Common Issues

1. **Metrics computation**: The metrics modules already handle edge cases like insufficient samples (<10) and return sensible defaults (0.0) with warnings
2. **Tensor device mismatches**: Use `validate_same_device()` or `ensure_device()` utilities to prevent device errors
3. **Resume functionality**: Use `--resume` flag to continue interrupted ablation studies without re-running completed experiments
