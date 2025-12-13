# Diversity in Multi-Objective GFlowNets

A systematic study investigating what drives diversity in Multi-Objective Generative Flow Networks (MOGFNs), featuring novel GFlowNet-specific diversity metrics and comprehensive factorial experiments across four benchmark tasks.

**Based on:** "Multi-Objective GFlowNets" by Jain et al. (ICML 2023) - https://proceedings.mlr.press/v202/jain23a.html

---

## Overview

This project explores how architectural choices, sampling strategies, and loss functions affect diversity in multi-objective optimization using GFlowNets. Through 405 factorial experiments and rigorous validation, we identify task-specific optimal configurations and demonstrate that **diversity is highly dependent on hyperparameter tuning**.

### Key Findings

- **Temperature is task-specific:** HyperGrid requires τ=5.0, Molecules requires τ=1.0, N-grams requires τ=2.0
- **Smaller models often perform better:** 64×2 architectures prevent mode collapse on molecular/discrete tasks
- **Trajectory Balance consistently outperforms alternatives** for quality-diversity balance
- **Novel QDS metric discovered a bug:** Hypervolume calculation inflated scores ~4× for 3D objectives (fixed)

---

## Environments

Four benchmark tasks with varying characteristics:

| Environment | State Space | Objectives | Key Challenge |
|-------------|-------------|------------|---------------|
| **HyperGrid** | 32×32 grid | 2 (corners) | Spatial exploration, mode collapse |
| **Molecules** | Fragment assembly | 3 (QED, SA, logP) | Chemical validity, precision |
| **N-grams** | 8-char sequences | 16 (bigrams) | Discrete combinatorics |
| **Sequences** | 20-base RNA | 3 (energy, pairs, length) | High variance, complex structure |

---

## Metrics

Implemented **4 categories** of diversity metrics (10 total):

### 1. Traditional (from MOEA/D, NSGA-II)
- Hypervolume, R2 Indicator, Spacing, Spread

### 2. GFlowNet-Specific
- **Trajectory Diversity Score (TDS):** Path uniqueness in state space
- **Mode Coverage Entropy (MCE):** DBSCAN-based spatial diversity
- **Quality-Diversity Score (QDS):** Composite metric (fixed bug: now uses correct HV)

### 3. Preference-Aware
- **Pareto Front Smoothness (PFS):** Preference alignment quality
- **Preference-Aligned Spread (PAS):** Diversity along preference directions

---

## Experiments

### Factorial Studies (405 total experiments)

Three 2-way factorial designs per task (135 experiments × 3 tasks):

1. **Capacity × Sampling:** Model size (small/medium/large) × Temperature (low/high/very high)
2. **Capacity × Loss:** Model size × Loss function (TB/SubTB/SubTB+entropy)
3. **Sampling × Loss:** Temperature × Loss function

**All experiments:** 5 random seeds, full metrics suite

### Validation

Best configurations identified by **average QDS across seeds** (not cherry-picked):

| Task | Config | Architecture | Temperature | QDS | MCE | Status |
|------|--------|--------------|-------------|-----|-----|--------|
| **N-grams** | small_tb | 64×2 | τ=2.0 | 0.576±0.005 | 0.606 | ✅ Validated (+2.3% vs factorial) |
| **Molecules** | small_low_tb | 64×2 | τ=1.0 | 0.165±0.001* | 0.294 | ✅ Validated (matched factorial) |
| **Sequences** | medium_veryhigh_tb | 64×3 | τ=5.0 | 0.219±0.014* | 0.439 | ✅ Validated (-5% vs factorial) |
| **HyperGrid** | medium_veryhigh_tb | 64×3 | τ=5.0 | 0.567±0.066 | 0.274 | ⚠️ Mode collapse (1/5 seeds) |

*QDS values corrected after fixing hypervolume bug

---

## Project Structure

```
diversity-mogfn/
├── src/
│   ├── environments/        # HyperGrid, Molecules, Sequences, N-grams
│   ├── models/              # MOGFN-PC, conditioning mechanisms
│   ├── metrics/             # 7 categories of diversity metrics
│   └── utils/               # Tensor/NumPy conversion utilities
├── configs/
│   ├── factorials/          # Factorial experiment configs
│   └── ablations/           # Legacy ablation configs
├── scripts/
│   ├── factorials/          # Per-task factorial experiment runners
│   ├── ablations/           # Analysis and plotting scripts
│   └── baselines/           # Baseline comparisons
└── results/
    ├── factorials/          # 405 factorial experiments
    └── validation/          # Best config validation runs
```

---

## Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/diversity-mogfn.git
cd diversity-mogfn

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Validation Experiments

```bash
# N-grams (best performing)
python scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_best_config.yaml \
    --output_dir results/validation/ngrams_best

# Molecules (with corrected QDS calculation)
python scripts/factorials/molecules/run_factorial_molecules.py \
    --config configs/factorials/molecules_best_config.yaml \
    --output_dir results/validation/molecules_best

# Sequences (corrected config: medium_veryhigh_tb)
python scripts/factorials/sequences/run_factorial_sequences.py \
    --config configs/factorials/sequences_best_config.yaml \
    --output_dir results/validation/sequences_best

# HyperGrid (note: mode collapse issue)
python scripts/factorials/hypergrid/run_factorial_hypergrid.py \
    --config configs/factorials/hypergrid_best_config.yaml \
    --output_dir results/validation/hypergrid_best
```

### Run Factorial Studies

```bash
# Example: Molecules capacity × sampling factorial
python scripts/factorials/molecules/run_factorial_molecules.py \
    --config configs/factorials/molecules_capacity_sampling_2way.yaml \
    --output_dir results/factorials/molecules_capacity_sampling
```

---

## Key Implementation Details

### Fixed QDS Bug (2025-12-12)

**Issue:** QDS calculation used incorrect hypervolume implementation for 3D objectives, inflating scores ~4×

**Fix:** Updated `src/metrics/composite.py` to use proper hypervolume from `traditional.py`

**Impact:** Molecules validation QDS changed from 0.455 (buggy) to 0.165 (correct), now matching factorial baseline

### Temperature Tuning is Critical

Different tasks require different exploration levels:
- **Molecules (τ=1.0):** Chemical validity requires precision
- **N-grams (τ=2.0):** Balanced discrete exploration
- **HyperGrid/Sequences (τ=5.0):** Broad exploration needed

**Lesson:** Temperature cannot be a fixed hyperparameter across tasks!

### Mode Collapse in HyperGrid

Spatial exploration tasks show high seed sensitivity:
- small_veryhigh_tb: 3/5 seeds collapsed to <5 modes
- medium_veryhigh_tb: 1/5 seeds collapsed
- **Open problem:** Requires further investigation

---

## Results Summary

### Validation Success Rate: 3/4 Tasks

✅ **N-grams:** Improved over factorial (+2.3% QDS, +9.4% MCE)
✅ **Molecules:** Matched factorial (after QDS fix), +38% MCE, 4× more modes
✅ **Sequences:** Nearly matched factorial (-5% MCE, -8% HV) with excellent stability
⚠️ **HyperGrid:** Mode collapse persists, needs investigation

### Cross-Seed Stability

Validated configurations show excellent generalization:
- N-grams: CV < 3% across all metrics
- Molecules: CV < 8% across all metrics
- Sequences: MCE CV = 1.3% (exceptionally stable)

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{diversity-mogfn-2025,
  author = {Demers, Katherine},
  title = {Diversity in Multi-Objective GFlowNets: A Systematic Study},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/diversity-mogfn}
}
```

Original MOGFN paper:
```bibtex
@inproceedings{jain2023multi,
  title={Multi-Objective GFlowNets},
  author={Jain, Moksh and Raparthy, Sharath Chandra and Hernandez-Garcia, Alex and others},
  booktitle={ICML},
  year={2023}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions or issues, please open a GitHub issue or contact katherine.demers@umontreal.ca.

**Last Updated:** December 12th 2025
