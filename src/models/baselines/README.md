# Baseline Algorithms for Multi-Objective Optimization

This module provides baseline algorithms for comparing MOGFN-PC performance against established multi-objective optimization methods.

## Implemented Baselines

### 1. Random Sampling (`random_sampler.py`)

**Description**: Samples random trajectories from the environment without any learning or optimization.

**Use case**: Simplest baseline to validate that learned methods outperform random search.

**API**:
```python
from src.models.baselines import RandomSampler
from src.environments.hypergrid import HyperGrid

env = HyperGrid(height=8, num_objectives=2)
sampler = RandomSampler(env, max_steps=100, seed=42)

# Sample trajectories
history = sampler.train(num_iterations=1000, batch_size=32)

# Get results
objectives = sampler.get_all_objectives()
pareto_front = sampler.get_pareto_front()
```

**Parameters**:
- `max_steps`: Maximum trajectory length
- `seed`: Random seed for reproducibility

**Metrics applicable**: Traditional, Spatial, Objective, Composite (not Trajectory/Flow/Dynamics)

---

### 2. NSGA-II (`nsga2_adapter.py`)

**Description**: Classic multi-objective genetic algorithm using non-dominated sorting and crowding distance for diversity.

**Dependencies**: Requires `pymoo` library
```bash
pip install pymoo
```

**Use case**: Standard baseline from evolutionary multi-objective optimization literature.

**API**:
```python
from src.models.baselines import NSGA2Adapter
from src.environments.hypergrid import HyperGrid

env = HyperGrid(height=8, num_objectives=2)
nsga2 = NSGA2Adapter(env, pop_size=100, max_steps=100, seed=42)

# Run NSGA-II
history = nsga2.train(num_iterations=50)  # 50 generations

# Get results
objectives = nsga2.get_all_objectives()
pareto_front = nsga2.get_pareto_front()
```

**Parameters**:
- `pop_size`: Population size (default: 100)
- `max_steps`: Maximum trajectory length
- `seed`: Random seed for reproducibility

**Metrics applicable**: Traditional, Spatial, Objective, Composite (not Trajectory/Flow/Dynamics)

**Note**: NSGA-II uses genetic operators (SBX crossover, PM mutation) optimized for continuous decision variables. Trajectories are encoded as continuous vectors that map to action selections.

---

## Running Baseline Comparisons

### Quick Test

```bash
# Test that baselines work
python tests/baselines/test_baselines_simple.py
```

### Full Comparison Script

```bash
# Compare Random + NSGA-II on HyperGrid (quick test)
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms random,nsga2 \
    --seeds 42 \
    --num_iterations 1000 \
    --output_dir results/baselines/test

# Full comparison with multiple seeds and HN-GFN
python scripts/baselines/run_baseline_comparison.py \
    --task hypergrid \
    --algorithms random,nsga2,hngfn \
    --seeds 42,153,264,375,486 \
    --num_iterations 10000 \
    --pop_size 100 \
    --output_dir results/baselines/hypergrid
```

**Arguments**:
- `--task`: Environment to use (`hypergrid`, `ngrams`, `molecules`, `sequences`)
- `--algorithms`: Comma-separated list of algorithms (`random`, `nsga2`, `hngfn`)
- `--seeds`: Comma-separated list of random seeds
- `--num_iterations`: Training iterations (Random/HN-GFN) or generations (NSGA-II)
- `--batch_size`: Batch size for Random/HN-GFN sampling (default: 32)
- `--pop_size`: Population size for NSGA-II (default: 100)
- `--output_dir`: Directory to save results

**Output Structure**:
```
results/baselines/hypergrid/
├── random_seed42/
│   ├── metrics.json
│   ├── objectives.npy
│   └── training_history.json
├── nsga2_seed42/
│   ├── metrics.json
│   ├── objectives.npy
│   ├── pareto_front.npy
│   └── training_history.json
├── hngfn_seed42/
│   ├── metrics.json
│   ├── objectives.npy
│   ├── pareto_front.npy
│   ├── checkpoint.pt
│   └── training_history.json
├── random_results.csv
├── nsga2_results.csv
├── hngfn_results.csv
├── all_results.csv
└── summary_by_algorithm.csv
```

---

## Metrics Computed

All baselines use the same 7-category metric framework:

1. **Traditional**: Hypervolume, Spacing, GD, IGD (if Pareto front known)
2. **Spatial**: MCE, PMD, PFS, num_unique_solutions
3. **Objective**: PAS (simplified approximation)
4. **Composite**: QDS, DER

**Note**: Trajectory, Flow, and Dynamics metrics require GFlowNet-specific data (trajectory samples, flow values, replay buffer) and are not applicable to Random/NSGA-II baselines.

---

## Comparison with MOGFN-PC

To compare your diversity metrics approach against these baselines:

### 1. Run baselines on all tasks
```bash
for task in hypergrid ngrams molecules sequences; do
    python scripts/baselines/run_baseline_comparison.py \
        --task $task \
        --algorithms random,nsga2,hngfn \
        --seeds 42,153,264,375,486 \
        --num_iterations 10000 \
        --output_dir results/baselines/$task
done
```

### 2. Run MOGFN-PC on same tasks
Use your existing ablation study infrastructure with matched computational budgets.

### 3. Compare metrics
- **Hypervolume**: Primary metric for Pareto front quality
- **Spacing**: Distribution uniformity
- **MCE**: Mode coverage and diversity
- **QDS**: Combined quality-diversity score

### 4. Statistical analysis
Use the CSV outputs to run t-tests, ANOVA, or other statistical comparisons across algorithms and seeds.

---

### 3. Hypernetwork-GFlowNet (`hn_gfn.py`)

**Description**: Uses a preference-conditioned hypernetwork for the log partition function Z instead of a fixed learned parameter.

**Dependencies**: Requires PyTorch (already installed)

**Use case**: State-of-the-art baseline from NeurIPS 2023 that improves sample efficiency in multi-objective settings.

**API**:
```python
from src.models.baselines import HN_GFN
from src.environments.hypergrid import HyperGrid

env = HyperGrid(height=8, num_objectives=2)
state_dim = env.state_dim  # State dimension (2 for HyperGrid coordinates)
num_actions = env.num_actions  # Number of actions (3 for HyperGrid: right, up, done)

hngfn = HN_GFN(
    env=env,
    state_dim=state_dim,
    num_objectives=env.num_objectives,
    hidden_dim=64,
    num_actions=num_actions,
    num_layers=3,
    z_hidden_dim=64,
    z_num_layers=3,
    learning_rate=1e-3,
    z_learning_rate=1e-3,
    alpha=1.5,
    seed=42
)

# Train HN-GFN
history = hngfn.train(num_iterations=1000, batch_size=32)

# Sample solutions
objectives, states = hngfn.sample(num_samples=100)

# Get results
all_objectives = hngfn.get_all_objectives()
pareto_front = hngfn.get_pareto_front()
```

**Parameters**:
- `state_dim`: Dimension of state space
- `num_objectives`: Number of objectives
- `hidden_dim`: Hidden dimension for policy networks (default: 64)
- `num_actions`: Number of possible actions
- `num_layers`: Layers in policy networks (default: 3)
- `z_hidden_dim`: Hidden dimension for Z hypernetwork (default: 64)
- `z_num_layers`: Layers in Z hypernetwork (default: 3)
- `learning_rate`: Learning rate for policy (default: 1e-3)
- `z_learning_rate`: Learning rate for Z hypernetwork (default: 1e-3)
- `alpha`: Dirichlet concentration parameter (default: 1.5)
- `max_steps`: Maximum trajectory length (default: 100)
- `seed`: Random seed for reproducibility

**Key Innovation**:
- **MOGFN-PC**: Uses fixed `log_Z` parameter (scalar)
- **HN-GFN**: Uses hypernetwork `Z(preference)` that outputs preference-dependent log partition function

This allows the partition function to adapt based on the preference vector, improving multi-objective optimization.

**Metrics applicable**: All metrics (Traditional, Trajectory, Spatial, Objective, Dynamics, Flow, Composite)

**Reference**:
Zhu et al. "Sample-efficient Multi-objective Molecular Optimization with GFlowNets" (NeurIPS 2023)
- Paper: https://arxiv.org/abs/2302.04040
- Code: https://github.com/violet-sto/HN-GFN

---

## Future Baselines (TODOs)

### 4. Single-Objective GFlowNet
Train separate GFlowNet for each objective, combine Pareto fronts.

**Advantage**: Tests if preference-conditioning is better than training per-objective.

---

## Implementation Notes

### Environment API Compatibility
All baselines work with the `MultiObjectiveEnvironment` interface:
- `get_initial_state()`: Get starting state
- `step(state, action)`: Returns `(next_state, done)`
- `get_valid_actions(state)`: Get valid actions
- `compute_objectives(state)`: Get objective values (returns PyTorch tensor)

### Tensor Handling
- Baselines convert PyTorch tensors to NumPy arrays for metric computation
- All metrics operate on NumPy arrays
- States remain as tensors throughout trajectory sampling

### Reference Points
For hypervolume computation, use:
```python
reference_point = np.array([1.1] * env.num_objectives)
```

This assumes objectives are normalized to [0, 1] range (as in HyperGrid).

---

## Citation

If using these baselines in research:

**NSGA-II**:
```bibtex
@article{deb2002fast,
  title={A fast and elitist multiobjective genetic algorithm: NSGA-II},
  author={Deb, Kalyanmoy and Pratap, Aravind and Agarwal, Sameer and Meyarivan, TAMT},
  journal={IEEE transactions on evolutionary computation},
  volume={6},
  number={2},
  pages={182--197},
  year={2002}
}
```

**MOGFN-PC** (comparison baseline):
```bibtex
@inproceedings{jain2023multi,
  title={Multi-Objective GFlowNets},
  author={Jain, Moksh and Raparthy, Sharath Chandra and Hern{\'a}ndez-Garc{\'i}a, Alex and others},
  booktitle={ICML},
  year={2023}
}
```

**HN-GFN** (Hypernetwork-GFlowNet):
```bibtex
@inproceedings{zhu2023sample,
  title={Sample-efficient Multi-objective Molecular Optimization with {GF}low{N}ets},
  author={Zhu, Yiheng and Wang, Kai and Wang, Zhongkai and Zhang, Zhe and Wang, Yue and Wu, Yining and others},
  booktitle={NeurIPS},
  year={2023}
}
```
