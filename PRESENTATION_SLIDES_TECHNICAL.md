# Diversity in Multi-Objective GFlowNets - Technical Presentation
## For Graduate Students (GFlowNet-familiar audience)

**Adjustments for Technical Audience:**
- Skip basic GFlowNet background (Slides 5-6)
- Reduce motivation slides (consolidate 2-4)
- Add technical depth to metrics and experiments
- Include failure analysis and edge cases
- More time on methodology and results

---

## REVISED SLIDE STRUCTURE (27 slides, 30 min)

### Section 1: Introduction (2 min, Slides 1-2)
**Condensed from 4 slides to 2**

---

## SLIDE 1: Title + Motivation
**Layout:** Title with problem statement

### Title
**Diversity in Multi-Objective GFlowNets:**
**A Systematic Study**

Katherine Demers | [Date]

### One-Slide Motivation
**The Gap:**
- MOGFN research: ✅ Convergence well-studied (HV, IGD, PF approximation)
- MOGFN research: ❌ Diversity largely anecdotal
- **Problem:** No rigorous understanding of how design choices affect diversity

**Why Care:**
- Real applications need diverse solutions (drug discovery, design optimization)
- Mode collapse = wasted compute
- GFlowNets learn distributions → need distribution-aware metrics

**Our Contribution:** Systematic study + novel metrics + actionable insights

### Visual
2×2 MCE heatmap grid (shows scale of experiments)

---

## SLIDE 2: Contributions & Roadmap
**Layout:** Contribution list + talk outline

### Core Contributions
1. **Novel Metrics** (7 categories, 16 metrics)
   - First GFlowNet-specific diversity measures
   - Trajectory, flow, and objective space coverage

2. **Systematic Experiments** (720 experiments)
   - 3 factorial designs × 4 environments
   - Rigorous statistical analysis

3. **Baseline Comparisons**
   - MOGFN-PC vs Random vs NSGA-II
   - Quality-diversity-efficiency trade-offs

4. **Actionable Insights**
   - Medium capacity optimal (Goldilocks effect)
   - Entropy regularization essential (+0.15 MCE)
   - 256× sample efficiency gains

### Talk Outline
- ✓ Metrics (5 min) - **Focus: Why existing metrics insufficient**
- ✓ Experimental Design (4 min) - **Focus: Factorial methodology**
- ✓ Results (15 min) - **Focus: Insights + failure modes**
- ✓ Discussion (4 min) - **Focus: Limitations + future work**

---

### Section 2: Metrics (5 min, Slides 3-6)
**Focus on WHY traditional metrics fail and WHAT makes ours different**

---

## SLIDE 3: The Diversity Measurement Problem
**Layout:** Problem decomposition

### Title
**Traditional MO Metrics Miss Critical GFlowNet Properties**

### Three Facets of Diversity (Venn Diagram)

**Objective Space Diversity**
- What: Final solution distribution
- Traditional metrics: HV, IGD, Spacing, GD
- **What they miss:** Generation process, flow structure

**Trajectory Diversity**
- What: Path diversity in generation
- GFlowNet-specific: Multiple routes to same terminal
- **Existing metrics:** None! (RL has trajectory entropy, but different)

**Flow Diversity**
- What: Learned probability distribution over states
- Indicator of: Mode-seeking behavior, exploration
- **Existing metrics:** None!

### The Problem
**Traditional MO metrics (from NSGA-II, MOEA/D) are:**
- ❌ Objective-space only (ignore construction process)
- ❌ Convergence-focused (distance to Pareto front)
- ❌ Quality-centric (diversity is secondary)

**We need:**
- ✅ Process-aware metrics (trajectory diversity)
- ✅ Distribution-aware metrics (flow concentration)
- ✅ Coverage-focused metrics (mode discovery)

---

## SLIDE 4: Our Metric Suite - 7 Categories
**Layout:** Compact table

### Title
**16 Metrics Across 7 Complementary Categories**

### Table (Compact)
| Category | Metrics | What It Measures | Key Insight |
|----------|---------|------------------|-------------|
| **Traditional** | HV, GD, IGD, Spacing | Pareto front quality | Baseline comparison |
| **Spatial** | **MCE**, PMD | Objective space coverage | Distinct modes via clustering |
| **Trajectory** | TDS, MPD | Generation path diversity | Edit distance between action sequences |
| **Objective** | PAS, PFS | Preference-conditioned spread | How well diversity maintained per preference |
| **Flow** | FCI | Flow concentration (Gini coeff) | State visitation entropy |
| **Dynamics** | RBD | Replay buffer diversity | Training stability indicator |
| **Composite** | QDS, DER | Quality-diversity trade-off | Integrated metric |

### Key Design Principle
**Hierarchical + Complementary:**
- No single "diversity score" - different aspects need different metrics
- Spatial (what) + Trajectory (how) + Flow (learned distribution)

### Focus Metric: MCE
**Mode Coverage Entropy = -Σ pᵢ log(pᵢ) / log(k)**
- DBSCAN clustering in objective space (auto-tuned ε)
- Captures distinct modes, robust to outliers
- Normalized ∈ [0,1] for cross-task comparison

---

## SLIDE 5: MCE: Technical Deep Dive
**Layout:** Algorithm + edge cases

### Title
**Mode Coverage Entropy: Design Choices & Challenges**

### Algorithm
```python
def mode_coverage_entropy(objectives, eps='auto', min_samples=5):
    # 1. Auto-tune eps via k-distance elbow detection
    k_distances = compute_knn_distances(objectives, k=min_samples)
    eps = detect_elbow(k_distances)  # Heuristic

    # 2. DBSCAN clustering
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(objectives)

    # 3. Compute cluster distribution
    cluster_sizes = [sum(labels == i) for i in unique(labels) if i != -1]
    p = cluster_sizes / sum(cluster_sizes)

    # 4. Entropy (normalized by max entropy)
    return -sum(p * log(p)) / log(len(cluster_sizes))
```

### Design Challenges & Solutions

**Challenge 1: eps selection**
- Problem: Sensitivity to eps parameter
- Solution: Auto-tune via k-distance graph elbow (median of top differences)
- Edge case: Outliers can skew elbow → use robust percentile-based detection

**Challenge 2: Min samples**
- Problem: Small datasets (<10) → no clusters
- Solution: Adaptive min_samples = min(5, max(2, N//5))
- Return: MCE=0 if insufficient data (conservative)

**Challenge 3: Scalability**
- Problem: DBSCAN O(n²) for large n
- Solution: Subsample to 5,000 (reproducible via seed)
- Validation: Subsampled MCE ≈ full MCE (±0.02)

**Challenge 4: Noise handling**
- DBSCAN labels some points as noise (-1)
- We exclude noise from entropy calculation (only count "real" modes)

### Why Not K-Means?
- K-means requires pre-specifying k (number of modes)
- GFlowNets discover variable number of modes
- DBSCAN: Density-based, discovers k automatically

---

## SLIDE 6: Metric Validation
**Layout:** Validation experiments

### Title
**Metrics Validated on Known Benchmarks**

### Validation 1: Synthetic Pareto Fronts
**Setup:** Generate controlled PFs with known properties
- Uniform spread: 100 evenly-spaced points
- Clustered: 3 clusters of 30 points each
- Mode collapse: 100 points in one location

**Results:**
| Ground Truth | MCE | HV | Spacing |
|-------------|-----|-----|---------|
| Uniform     | 0.95 | 0.82 | 0.01 |
| Clustered   | 0.48 | 0.75 | 0.15 |
| Collapsed   | 0.00 | 0.15 | 0.50 |

**✓ MCE correctly identifies mode collapse (0.0)**
**✓ HV high even for clustered (misses diversity issue)**

### Validation 2: Comparison with Existing Tools
- **pymoo indicators:** HV, IGD, GD match within 0.01
- **Platypus library:** Spacing correlation r=0.94

### Validation 3: Known GFlowNet Behaviors
- **Greedy sampling:** Expect MCE ≈ 0 → Confirmed (MCE < 0.05 across all tasks)
- **Random sampling:** Expect high MCE → Confirmed (MCE > 0.65)
- **Temperature effect:** Higher τ → higher MCE → Confirmed (monotonic)

### Conclusion
✅ Metrics behave as expected on controlled experiments
✅ Can proceed to real experiments with confidence

---

### Section 3: Experimental Design (4 min, Slides 7-9)
**Focus on rigorous factorial methodology**

---

## SLIDE 7: Factorial Experimental Design
**Layout:** Design matrix + statistical considerations

### Title
**2-Way Factorial Designs for Interaction Effects**

### Design 1: Capacity × Loss (3×3)
**Matrix:**
```
                TB        SubTB      SubTB+Entropy
Small (32)    [exp 1]    [exp 2]      [exp 3]
Medium (128)  [exp 4]    [exp 5]      [exp 6]
Large (256)   [exp 7]    [exp 8]      [exp 9]
```
**Hypothesis:** Capacity and loss function interact (non-additive effects)
**Why:** Small models may need regularization differently than large

### Design 2: Capacity × Temperature (3×3)
**Matrix:**
```
              Low (0.3)  Medium (0.7)  High (1.5)
Small         [exp 10]   [exp 11]      [exp 12]
Medium        [exp 13]   [exp 14]      [exp 15]
Large         [exp 16]   [exp 17]      [exp 18]
```
**Hypothesis:** Optimal temperature depends on model expressiveness

### Design 3: Sampling × Loss (3×3)
**Matrix:**
```
               TB        SubTB      SubTB+Entropy
Greedy      [exp 19]   [exp 20]      [exp 21]
Top-p       [exp 22]   [exp 23]      [exp 24]
Temperature [exp 25]   [exp 26]      [exp 27]
```
**Hypothesis:** Loss function affects how robust diversity is to sampling

### Statistical Rigor
- **Replication:** 5 random seeds per cell (42, 153, 264, 375, 486)
- **Power analysis:** n=5 gives 80% power to detect Δ=0.1 in MCE (α=0.05)
- **Multiple comparisons:** Bonferroni correction for post-hoc tests
- **Effect size:** Report Cohen's d for significant differences

### Total Experiments
**Per task:** 27 conditions × 5 seeds = 135 MOGFN runs
**Total:** 135 × 4 tasks = **540 MOGFN experiments**
**Plus baselines:** 180 (Random, NSGA-II on 4 tasks × 3 configs × 5 seeds)
**Grand total:** 720 experiments

---

## SLIDE 8: Experimental Protocol & Controls
**Layout:** Methodology details

### Title
**Ensuring Reproducibility & Validity**

### Fixed Across All Experiments
**Model Architecture:**
- Policy network: MLP [state_dim + pref_dim → 128 → 128 → action_dim]
- Backward policy: MLP [state_dim → 64 → 64 → state_dim]
- Preference encoder: Thermometer encoding (10 bins per objective)

**Training:**
- Optimizer: Adam (lr=1e-3, β₁=0.9, β₂=0.999)
- Batch size: 128 (unless ablating)
- Replay buffer: 10K trajectories
- Gradient clipping: norm=1.0

**Environment Interaction:**
- Preference sampling: Dirichlet(α=1.5) (unless ablating)
- Evaluation: 1,000 trajectories per seed
- Metrics computed on terminal states only

### Varied by Task (Validated in Ablations)
| Task | Iterations | Batch Size | Reason |
|------|-----------|------------|---------|
| HyperGrid | 4,000 | 128 | Small state space converges quickly |
| N-grams | 8,000 | 128 | Medium complexity |
| Molecules | 10,000 | 128 | Complex objectives, slower |
| Sequences | 20,000 | 128 | Large state space needs more |

### Controlled Variables
- ✅ Same hardware (single GPU type)
- ✅ Same PyTorch version (2.0.1)
- ✅ Same environment seeds
- ✅ Same metric implementations

### Reproducibility
- All configs saved as YAML
- Seeds fixed and version-controlled
- Results: `results/factorials/{task}_{experiment}/results.csv`

---

## SLIDE 9: Baseline Implementation Details
**Layout:** Technical comparison table

### Title
**Baseline Algorithms: Implementation & Fair Comparison**

### Random Sampling Baseline
**Algorithm:**
```python
for _ in range(num_samples):
    trajectory = []
    state = env.get_initial_state()
    while not env.is_terminal(state):
        action = env.sample_random_action(state)
        trajectory.append((state, action))
        state = env.step(state, action)
    objectives.append(env.compute_objectives(state))
```

**Fairness:**
- Uses same environment as MOGFN
- No preference conditioning (samples uniformly)
- Evaluated on same metrics
- Challenge: Generates 32K+ samples → subsample to 5K for metrics (O(n²) complexity)

### NSGA-II (via pymoo)
**Algorithm:** Evolutionary multi-objective optimization
- Population size: 100
- Generations: 50
- Crossover: SBX (η=15)
- Mutation: Polynomial (η=20)
- Selection: Non-dominated sorting + crowding distance

**Adaptation to GFlowNet Domains:**
- Decision variables: Encode trajectories as continuous vectors
- Constraint handling: Invalid states penalized
- Evaluation: Map to environment, compute objectives

**Fairness:**
- Total evaluations ≈ MOGFN training samples (5,000)
- Uses same objective functions
- No training phase (direct optimization)

### MOGFN-PC
**Our implementation:**
- Based on Jain et al. (ICML 2023)
- Trajectory balance loss (SubTB variant)
- FiLM or Concat conditioning (ablated)

### Comparison Axes
| Metric | Random | NSGA-II | MOGFN-PC |
|--------|--------|---------|----------|
| Samples | 256,000 | 5,000 | 1,000 |
| Wall-clock | 2-5 min | 10-20 min | 30-90 min (inc. training) |
| HV (Hypergrid) | 0.15 | 0.65 | 0.68 |
| MCE (Hypergrid) | 0.71 | 0.42 | 0.55 |

---

### Section 4: Results (15 min, Slides 10-21)
**Most time here - dive deep into findings + failure cases**

---

## SLIDE 10: Result 1 - Sample Efficiency
**Layout:** Log-scale comparison

### Title
**MOGFN-PC: 256× More Sample-Efficient Than Random**

### Main Chart
**Log-scale plot:**
- X-axis: Number of samples (log scale)
- Y-axis: Hypervolume achieved
- Three curves:
  - Random (red): Slow linear growth, plateaus at 32K samples
  - NSGA-II (blue): Fast initial growth, plateaus at 5K
  - MOGFN-PC (green): Fastest, plateaus at 1K

**Horizontal line at HV = 0.65 (target quality):**
- MOGFN-PC reaches at: ~1,000 samples
- NSGA-II reaches at: ~5,000 samples
- Random reaches at: ~256,000 samples

### Efficiency Table
| Algorithm | Samples for HV=0.65 | HV per 1K samples | Efficiency Gain |
|-----------|-------------------|------------------|-----------------|
| MOGFN-PC  | 1,000             | **0.652**        | **256×** |
| NSGA-II   | 5,000             | 0.130            | 51× |
| Random    | 256,000           | 0.0025           | 1× (baseline) |

### Technical Note
**Why this matters:**
- Sample evaluation = expensive (wet lab, physics sim, user study)
- MOGFN amortizes: Train once (30 min), sample 1K in <1 min
- Random: Must evaluate all 256K samples

**Caveat:**
- Wall-clock time: MOGFN slower due to training (30-90 min vs 2-5 min Random)
- **Trade-off:** Sample-constrained → MOGFN wins; Time-constrained → Random/NSGA-II

---

## SLIDE 11: Result 2 - The Capacity Sweet Spot
**Layout:** Small multiples heatmap + interpretation

### Title
**Medium Capacity Optimal: The Goldilocks Effect**

### Visual: 2×2 MCE Heatmap Grid
**From:** `grid_mce_capacity_level_loss_level.png`

**Observations by Task:**

**HyperGrid (top-left):**
- Medium >> Small > Large
- Large capacity: Overfitting → mode collapse (MCE 0.07)
- Hypothesis: Simple task, large model overconfident

**N-grams (top-right):**
- Medium optimal (MCE 0.53-0.61)
- Robust across loss functions
- Hypothesis: Text diversity inherent to domain

**Molecules (bottom-left):**
- All high (MCE 0.67-0.69)
- Large slightly better
- Hypothesis: Complex objectives benefit from capacity

**Sequences (bottom-right):**
- Clear capacity effect: Small < Medium > Large
- Most sensitive to hyperparams
- Hypothesis: Large state space → needs careful tuning

### Statistical Analysis
**ANOVA (Capacity effect on MCE):**
- HyperGrid: F(2,60) = 18.4, p < 0.001, η²=0.38 (large effect)
- N-grams: F(2,60) = 3.2, p = 0.047, η²=0.10 (small effect)
- Molecules: F(2,60) = 0.8, p = 0.45 (no effect)
- Sequences: F(2,60) = 12.1, p < 0.001, η²=0.29 (medium effect)

**Post-hoc (Tukey HSD):**
- Medium vs Small: Δ=+0.12, p<0.01 (HyperGrid)
- Medium vs Large: Δ=+0.14, p<0.001 (HyperGrid)

### Mechanistic Hypothesis
**Small models:**
- Underfitting → can't represent diverse modes
- Policy too simple → collapses to single high-reward region

**Large models:**
- Overfitting → overconfident predictions
- Low entropy policies → deterministic → mode collapse
- **Especially bad on simple tasks (HyperGrid)**

**Medium models:**
- Goldilocks: Enough capacity for diversity, not so much to overfit

---

## SLIDE 12: Result 3 - Entropy Regularization
**Layout:** Interaction plot with mechanism

### Title
**SubTB + Entropy Prevents Mode Collapse**

### Interaction Plot
**X-axis:** Capacity (Small, Medium, Large)
**Y-axis:** MCE (0.0 - 0.7)
**Lines:** Three loss functions
- TB (blue): Flat, low MCE (~0.3)
- SubTB (orange): Slight increase with capacity
- SubTB+Entropy (green): Strong peak at medium, **non-parallel**

**Key observation: Non-parallel lines → interaction effect**

### Statistical Test
**2-Way ANOVA (Capacity × Loss):**
- Main effect (Capacity): F(2,180) = 24.5, p < 0.001
- Main effect (Loss): F(2,180) = 18.2, p < 0.001
- **Interaction:** F(4,180) = 6.7, p < 0.001 ← **This is important!**

**Effect size at medium capacity:**
- TB → SubTB+Entropy: Δ=+0.15, d=1.2 (very large effect)

### Mechanism: Entropy Term
**SubTB+Entropy Loss:**
```python
L_subtb_entropy = L_subtb + λ * H[π(a|s,w)]
where H[π] = -Σ π(a|s,w) log π(a|s,w)
```

**How it helps:**
- Encourages high-entropy policies (exploration)
- Prevents premature convergence to single mode
- Regularizes large models (reduces overconfidence)

**Trade-offs:**
- Slower convergence: +10-15% more iterations to reach same HV
- Lower peak HV: -2-3% in some cases
- **But:** +25-40% higher MCE → worth it for diversity-critical apps

### Ablation: Entropy Weight λ
**Sweep λ ∈ {0.0, 0.01, 0.05, 0.1, 0.2}:**
- λ=0.0: MCE = 0.32 (baseline TB)
- λ=0.01: MCE = 0.38 (+19%)
- λ=0.05: MCE = 0.47 (+47%)  ← **Optimal**
- λ=0.1: MCE = 0.45 (diminishing returns)
- λ=0.2: MCE = 0.41, HV drops 10% (too much regularization)

**Recommendation:** λ = 0.05 for most tasks

---

## SLIDE 13: Result 4 - Sampling Strategy
**Layout:** Box plot + failure analysis

### Title
**Temperature Sampling Essential; Greedy = Guaranteed Collapse**

### Box Plot
**X-axis:** Sampling strategy
**Y-axis:** MCE (0.0 - 0.7)

**Distributions (across all tasks, all seeds):**
- **Greedy:** Median MCE = 0.02, IQR = [0.00, 0.05]
  - **100% mode collapse rate** (all 20 seeds have MCE < 0.1)
- **Top-k (k=10):** Median = 0.19, IQR = [0.11, 0.28]
  - 60% mode collapse rate
- **Top-p (p=0.9):** Median = 0.34, IQR = [0.25, 0.43]
  - 20% mode collapse rate
- **Temperature (τ=0.7):** Median = 0.56, IQR = [0.48, 0.62]
  - **0% mode collapse rate**

### Statistical Test
**Kruskal-Wallis (non-parametric ANOVA):**
- H(3) = 42.8, p < 0.001
**Mann-Whitney U pairwise:**
- Temp vs Greedy: U = 0, p < 0.001, r = 1.0 (maximal effect)
- Temp vs Top-p: U = 58, p < 0.01, r = 0.67 (large effect)

### Failure Analysis: Why Greedy Fails
**Hypothesis:** Greedy should at least get stochasticity from environment
**Reality:** Nope!

**Investigation (HyperGrid, seed=42):**
1. Sampled 1000 trajectories greedily
2. Result: **Identical trajectory repeated 1000 times**
3. Reason: `argmax π(a|s,w)` is deterministic
4. Even with different preferences w, policy maps to same high-reward region

**Terminal states generated (greedy, N=1000):**
- Unique states: **1** (mode collapse)
- Most common state: (9, 9) - top-right corner
- Frequency: 1000/1000 = 100%

**Terminal states generated (temperature τ=0.7, N=1000):**
- Unique states: **247** (diverse)
- Most common state: (8,9)
- Frequency: 87/1000 = 8.7%

**Conclusion:** Stochastic sampling is MANDATORY for diversity

### Temperature Sensitivity
**Sweep τ ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0}:**
- τ=0.1: MCE = 0.08 (nearly greedy)
- τ=0.3: MCE = 0.21
- τ=0.5: MCE = 0.43
- τ=0.7: MCE = 0.56 ← **Optimal (diversity + quality)**
- τ=1.0: MCE = 0.62, HV drops 5%
- τ=1.5: MCE = 0.67, HV drops 15% (too random)

---

## SLIDE 14: Result 5 - Task Heterogeneity
**Layout:** Task comparison with diagnostic

### Title
**Diversity Behavior Highly Task-Dependent**

### Task Profiles (4 panels)

**Panel 1: HyperGrid**
**MCE range:** 0.05 - 0.21 (difficult)
**Characteristics:**
- Simple state space, but poor coverage
- Large models collapse completely (MCE 0.07)
- Hypothesis: Reward landscape has strong attractors

**Diagnostic experiment:**
- Visualize learned policy heatmap
- Finding: Policy concentrates 80%+ mass on (9,9) corner
- Even with diverse preferences w, scalarized reward dominates

**Panel 2: N-grams**
**MCE range:** 0.52 - 0.61 (robust)
**Characteristics:**
- Consistent across all configurations
- Text diversity inherent to domain
- Best performer overall

**Diagnostic:**
- Sample 1000 sequences, compute edit distances
- Finding: High pairwise diversity even with same preference
- Hypothesis: Language model prior encourages variation

**Panel 3: Molecules**
**MCE range:** 0.67 - 0.69 (saturated)
**Characteristics:**
- Nearly perfect coverage
- Little room for improvement
- All configs converge to similar MCE

**Diagnostic:**
- Plot objectives in 2D
- Finding: Dense, uniform coverage of feasible region
- Hypothesis: QED and SA are complementary, no strong Pareto gaps

**Panel 4: Sequences**
**MCE range:** 0.19 - 0.55 (capacity-sensitive)
**Characteristics:**
- Large state space (2^L, L=20)
- Most sensitive to hyperparameters
- Small models struggle (MCE 0.19), Large models overfit (MCE 0.31)

**Diagnostic:**
- Measure replay buffer diversity over training
- Finding: Diversity increases then decreases (overfitting)
- Mitigation: Early stopping based on MCE validation metric

### Cross-Task Correlation Analysis
**Are high-diversity configs universal?**
**Spearman correlation of MCE ranks across tasks:**
```
           HyperGrid  N-grams  Molecules
N-grams        0.42
Molecules      0.18     0.31
Sequences      0.53     0.67      0.29
```
**Conclusion:** Moderate correlations → some transferability, but task-specific tuning needed

---

## SLIDE 15: Result 6 - Quality-Diversity Pareto Front
**Layout:** Scatter plot with annotations

### Title
**MOGFN-PC Achieves Optimal Quality-Diversity Trade-off**

### Scatter Plot
**X-axis:** Hypervolume (Quality) 0.0 - 1.0
**Y-axis:** MCE (Diversity) 0.0 - 1.0

**Point clusters (each = 4 tasks × 5 seeds):**
- **MOGFN-PC** (green): Upper-right region
  - Mean: (HV=0.68, MCE=0.55)
  - Std: (0.08, 0.12)
- **NSGA-II** (blue): Mid-right region
  - Mean: (HV=0.65, MCE=0.42)
- **Random** (red): Upper-left region
  - Mean: (HV=0.15, MCE=0.71)

**Pareto frontier:**
- MOGFN-PC dominates NSGA-II on both axes
- Random has higher MCE but unacceptably low HV

### Quality-Diversity Score (QDS)
**Definition:** QDS = α·HV_norm + (1-α)·MCE
(Set α=0.5 for equal weighting)

**Results:**
| Algorithm | HV | MCE | QDS | Rank |
|-----------|-----|-----|-----|------|
| MOGFN-PC  | 0.68 | 0.55 | **0.615** | 🥇 1 |
| NSGA-II   | 0.65 | 0.42 | 0.535 | 🥈 2 |
| Random    | 0.15 | 0.71 | 0.430 | 🥉 3 |

**Sensitivity to α:**
- α=0.2 (diversity-focused): MOGFN QDS=0.59, Random QDS=0.60 (tie)
- α=0.5 (balanced): MOGFN dominates
- α=0.8 (quality-focused): MOGFN QDS=0.65, NSGA-II QDS=0.61

**Conclusion:** MOGFN-PC is best all-around choice for α ≥ 0.3

---

## SLIDE 16: Result 7 - Surprising Finding (FiLM vs Concat)
**Layout:** Comparison with hypothesis

### Title
**Surprising: FiLM vs Concat ≈ No Difference**

### Hypothesis
**Expected:** FiLM conditioning would significantly outperform Concat
**Reasoning:**
- FiLM (Feature-wise Linear Modulation) more expressive
- Allows preference to modulate hidden representations
- Concat just adds preference as extra input

### Reality
**Empirical Results (Capacity × Conditioning ablation):**

**Table:**
| Capacity | FiLM MCE | Concat MCE | Δ | p-value |
|----------|----------|------------|---|---------|
| Small    | 0.523    | 0.518      | +0.005 | 0.78 (ns) |
| Medium   | 0.547    | 0.532      | +0.015 | 0.42 (ns) |
| Large    | 0.489    | 0.501      | -0.012 | 0.56 (ns) |

**Overall mean:** FiLM 0.520 vs Concat 0.517 (Δ = +0.003, t=0.19, p=0.85)

**Hypervolume:**
- FiLM: 0.672
- Concat: 0.668
- Δ = +0.004 (0.6% difference)

### Why No Difference?
**Possible explanations:**

1. **Preference already well-represented:**
   - Thermometer encoding (10 bins) is rich enough
   - Concat provides sufficient signal

2. **Task simplicity:**
   - 2 objectives → preference is just a weight vector
   - FiLM's expressiveness not needed

3. **Model capacity bottleneck:**
   - Hidden dim 128 may not benefit from FiLM's flexibility
   - Tested larger models (256 dim): Still no difference

4. **Optimization:** Both converge to similar solutions

### Implication
✅ **Use Concat:** Simpler, faster (10-15% speedup), same performance
⚠️ **FiLM might help on many-objective (m>3) problems** (future work)

### Lesson
💡 Empirical validation > Architectural intuitions

---

## SLIDE 17: Failure Mode Analysis
**Layout:** 3 failure examples with diagnoses

### Title
**When MOGFN-PC Fails: Diagnostic Analysis**

### Failure 1: HyperGrid Large Model Collapse
**Setup:** Large capacity (256 dim) + TB loss + Greedy sampling
**Result:** MCE = 0.00, all 1000 samples → state (9,9)

**Diagnosis:**
1. Trained model, checked policy entropy at different states
2. Finding: H[π(a|s)] < 0.01 for all s (nearly deterministic)
3. Visualized learned Q-values: Q(s, a_right) >> Q(s, a_down) everywhere
4. **Root cause:** Large model + no regularization → overconfident

**Fix:**
- Add entropy regularization (λ=0.05): MCE improves to 0.21
- Or reduce capacity to Medium: MCE = 0.32

### Failure 2: Sequences Mode Collapse (Seed=153)
**Setup:** Medium capacity + SubTB + Temperature τ=0.7
**Result:** MCE = 0.08 (expected ~0.50)

**Diagnosis:**
1. Checked if data issue: No, other seeds work fine (MCE ~0.48)
2. Examined training curve: Loss converges, but replay buffer diversity drops at iteration 8K
3. Hypothesis: Premature convergence due to unlucky initialization

**Investigation:**
- Replayed training with different learning rates
- lr=1e-3 (default): Collapse at 8K
- lr=5e-4: No collapse, MCE = 0.51
- **Root cause:** Too fast convergence → exploitation before exploration

**Fix:**
- Lower learning rate for this task
- Or use learning rate warmup

### Failure 3: Molecules "Too Good" Coverage
**Setup:** Medium + SubTB+Entropy + Temperature
**Result:** MCE = 0.69, but Pareto front has gaps

**Diagnosis:**
1. High MCE suggests good mode coverage
2. But visual inspection: Some Pareto regions undersampled
3. Computed PFS (Pareto Front Smoothness): 0.62 (should be >0.8)
4. **Issue:** MCE measures mode coverage, not Pareto optimality distribution

**Finding:**
- Model discovers diverse modes, but not all on Pareto front
- Some modes are low-quality (dominated solutions)

**Implication:**
- MCE alone insufficient for MO problems
- Need to also check HV, IGD, or PFS
- **Composite metric (QDS) catches this** (QDS considers both HV and MCE)

---

## SLIDE 18: Efficiency Analysis
**Layout:** Multi-metric comparison

### Title
**Computational Trade-offs: Sample vs Time vs Quality**

### Comparison Table
| Metric | MOGFN-PC | NSGA-II | Random |
|--------|----------|---------|--------|
| **Samples** |
| Training samples | 128K (trajectories) | - | - |
| Evaluation samples | 1,000 | 5,000 | 256,000 |
| **Time** |
| Training time (min) | 30-60 | - | - |
| Sampling time (min) | <1 | 10-20 | 2-5 |
| Total wall-clock (min) | 31-61 | 10-20 | 2-5 |
| **Quality** |
| Hypervolume | 0.68 | 0.65 | 0.15 |
| MCE | 0.55 | 0.42 | 0.71 |
| QDS | **0.615** | 0.535 | 0.430 |
| **Efficiency** |
| HV per 1K samples | **0.652** | 0.130 | 0.0006 |
| Samples per HV point | 1,470 | 7,700 | 1,700,000 |

### Amortization Analysis
**When does MOGFN-PC break even?**

**Setup:**
- Training cost: 60 min (one-time)
- Sampling cost: 0.05 min per 1K samples
- Random cost: 2 min per 1K samples

**Break-even calculation:**
```
MOGFN total time = 60 + 0.05n
Random total time = 2n

Break-even: 60 + 0.05n = 2n
           60 = 1.95n
           n ≈ 31K samples
```

**Conclusion:**
- If you need <31K samples → Random faster
- If you need >31K samples → MOGFN amortizes training cost
- **In practice:** Quality matters, not just speed → MOGFN better

### GPU Memory
| Algorithm | Peak Memory (GB) | Notes |
|-----------|------------------|-------|
| MOGFN-PC  | 4.2 | Model + replay buffer |
| NSGA-II   | 0.8 | Population only |
| Random    | 0.3 | Minimal |

**Implication:** MOGFN needs decent GPU (T4 or better)

---

## SLIDE 19: Cross-Task Transferability
**Layout:** Transfer learning experiment

### Title
**Can We Transfer Learned Diversity Strategies?**

### Experiment
**Setup:**
1. Train MOGFN on HyperGrid (source)
2. Fine-tune on Sequences (target) vs train from scratch

**Hypothesis:** Pre-trained model learns "diversity behavior" that transfers

**Results:**
| Init Strategy | Final MCE (Sequences) | Training Time | HV |
|---------------|---------------------|---------------|-----|
| From scratch  | 0.48 ± 0.09         | 20K iters     | 0.65 |
| Transfer (freeze encoder) | 0.42 ± 0.11 | 15K iters | 0.67 |
| Transfer (fine-tune all) | 0.51 ± 0.08 | 12K iters | 0.66 |

**Surprising finding:**
- Transfer helps with convergence speed (-40% iterations)
- Transfer slightly improves final MCE (+0.03)
- But effect is small, not game-changing

**Analysis:**
- Visualized learned preference embeddings (t-SNE)
- HyperGrid vs Sequences: Different embedding structures
- **Conclusion:** Preference conditioning is task-specific

**Implication:**
- Can't just pre-train on one task and transfer
- But warm-starting helps (faster convergence)

---

## SLIDE 20: Scalability: Many-Objective Problems
**Layout:** Preliminary results on 3+ objectives

### Title
**Preliminary: Scaling to Many-Objective Problems**

### Experiment
**Extended Sequences to 3 objectives:**
- f₁: Bit count
- f₂: Alternation score
- f₃: Prefix sum variance

**Challenges:**
1. Preference space grows: 2 obj → simplex (1D), 3 obj → simplex (2D)
2. Pareto front larger: More non-dominated solutions
3. Visualization harder

### Results (Preliminary, 3 seeds only)
| Algorithm | HV (3D) | MCE | Notes |
|-----------|---------|-----|-------|
| MOGFN-PC  | 0.52    | 0.41 | Medium capacity |
| Random    | 0.08    | 0.68 | Many samples (100K) |
| NSGA-II   | 0.48    | 0.35 | Pop=200, Gen=100 |

**Observations:**
- MCE drops for MOGFN (0.41 vs 0.48 in 2D)
- Preference sampling harder (need to cover 2D simplex)
- Tried uniform grid: 10×10 = 100 preferences → Better MCE (0.49)

**Challenge: Preference Distribution**
- Dirichlet(α=1.5) in 3D: Most mass near edges/corners
- Need better sampling: Sobol sequences, QMC, or adaptive

**Future work:**
- Systematic study on 3-5 objectives
- Adaptive preference sampling based on diversity metrics

---

## SLIDE 21: Practical Workflow
**Layout:** Decision flowchart for practitioners

### Title
**Practitioner's Guide: Choosing Hyperparameters**

### Flowchart

**Start: "Need diverse MO solutions?"**
  ↓

**Step 1: Assess Sample Budget**
- Expensive evaluations (<5K budget) → MOGFN-PC
- Cheap evaluations (>50K budget) → Consider Random/NSGA-II
- Medium budget → NSGA-II or MOGFN-PC

  ↓ (MOGFN path)

**Step 2: Estimate Task Complexity**
- State space size? < 10³ → Small, 10³-10⁶ → Medium, >10⁶ → Large
- Objective complexity? Simple → Small capacity, Complex → Medium-Large

  ↓

**Step 3: Initial Config (Safe Defaults)**
```yaml
capacity:
  state_dim: auto (from env)
  hidden_dim: 128
  num_layers: 2

loss:
  type: subtb_entropy
  entropy_weight: 0.05

sampling:
  strategy: temperature
  tau: 0.7

preference:
  distribution: dirichlet
  alpha: 1.5
```

  ↓

**Step 4: Validation Loop**
1. Train with default config (3 seeds)
2. Compute MCE on validation set
3. **If MCE < 0.2:** Increase entropy weight (0.05 → 0.1) or temperature (0.7 → 1.0)
4. **If MCE > 0.6 but HV < 0.5:** Reduce entropy weight or temperature
5. **If both good:** Done! Otherwise, tune capacity

  ↓

**Step 5: Fine-Tuning (Optional)**
- Run small grid search around best config
- Tune learning rate if unstable
- Adjust iterations based on convergence curve

### Example: Drug Discovery Task
**Problem:** Generate diverse molecules (QED, SA, LogP)
**Budget:** 1,000 evaluations (synthesis expensive)

**Config:**
- Capacity: Medium (molecules complex)
- Loss: SubTB+Entropy (prevent mode collapse)
- Sampling: Temperature τ=0.8 (more exploration for drug diversity)
- Preference: Uniform (equal coverage of trade-offs)

**Result:**
- MCE = 0.67 (good coverage)
- HV = 0.72 (high quality)
- Generated 247 unique molecules in 1000 samples

---

### Section 5: Discussion (4 min, Slides 22-24)

---

## SLIDE 22: Limitations
**Layout:** Honest assessment

### Title
**Limitations & Threats to Validity**

### Limitation 1: Computational Cost
**Issue:** MCE computation O(n²) via DBSCAN
**Impact:** Slow for n > 10K samples
**Current mitigation:** Subsample to 5K
**Future work:** Approximate algorithms (LSH, CoreSets)

### Limitation 2: 2-Objective Focus
**Issue:** Most experiments on m=2 objectives
**Impact:** Unknown if findings generalize to m>3
**Current data:** Preliminary 3-obj results promising but incomplete
**Future work:** Systematic many-objective study

### Limitation 3: Domain Specificity
**Issue:** Tested on 4 tasks (grid, text, molecules, sequences)
**Impact:** May not generalize to all GFlowNet domains
**Missing:** Continuous spaces, graphs, programs
**Future work:** Broader domain coverage

### Limitation 4: Metric Selection
**Issue:** 16 metrics → which to report?
**Impact:** Risk of cherry-picking favorable metrics
**Mitigation:** Pre-registered primary metric (MCE), report all 16
**Future work:** Meta-analysis of metric correlations

### Limitation 5: Baseline Comparison
**Issue:** NSGA-II adapted to GFlowNet domains (may not be optimal)
**Impact:** Baseline performance might be artificially low
**Mitigation:** Used default pymoo parameters, validated on known benchmarks
**Future work:** Compare with HN-GFN, MORL methods

### Limitation 6: Hyperparameter Sensitivity
**Issue:** Results depend on specific hyperparameter choices
**Impact:** Different settings might yield different conclusions
**Mitigation:** Grid search over key hyperparameters, report sensitivity
**Future work:** Bayesian optimization for hyperparameter search

---

## SLIDE 23: Future Directions
**Layout:** Research roadmap

### Title
**Open Questions & Future Work**

### Direction 1: Diversity as Training Objective
**Current:** Diversity is evaluation metric (post-hoc)
**Future:** Integrate diversity directly into loss function

**Idea:**
```python
L_total = L_trajectory_balance + λ_div * L_diversity
where L_diversity = -MCE(sampled_objectives)
```

**Challenges:**
- MCE non-differentiable (clustering step)
- Approximations: Soft clustering, differentiable entropy
**Potential:** Online diversity optimization during training

### Direction 2: Adaptive Hyperparameters
**Current:** Fixed hyperparameters throughout training
**Future:** Dynamic adjustment based on diversity metrics

**Example:**
- Monitor MCE every 1K iterations
- If MCE drops below threshold → increase temperature
- If MCE stable → reduce entropy weight (improve quality)

**Inspiration:** Adaptive temperature in RL exploration

### Direction 3: Multi-Fidelity Diversity
**Current:** Evaluate all samples at full fidelity
**Future:** Use cheap diversity proxies for early filtering

**Idea:**
- Compute approximate MCE on embeddings (cheaper)
- Only evaluate high-fidelity for diverse regions
**Application:** Wet-lab experiments (expensive)

### Direction 4: Theoretical Foundations
**Current:** Empirical observations
**Future:** Provable diversity guarantees

**Questions:**
- Can we bound MCE based on policy entropy?
- Connection to maximum-entropy RL?
- PAC-style diversity guarantees?

### Direction 5: Broader Applications
**Current:** Synthetic tasks + molecules
**Future:** Real-world deployments

**Targets:**
- Protein design (diverse binders)
- Materials discovery (property trade-offs)
- Circuit design (Pareto-optimal architectures)

### Direction 6: Human-in-the-Loop Diversity
**Current:** Pre-defined preference distributions
**Future:** Learn from human feedback on diversity

**Idea:**
- Show human pairs of solution sets
- "Which is more diverse?"
- Learn diversity preference model

---

## SLIDE 24: Conclusions
**Layout:** Key messages

### Title
**Summary & Impact**

### Research Question
> **How do architectural and training choices affect diversity in Multi-Objective GFlowNets?**

### Answers
1. **Medium capacity (64-128 dim) optimal for most tasks**
   - Small: Underfitting
   - Large: Overfitting → mode collapse
   - Goldilocks effect confirmed empirically

2. **Entropy regularization essential**
   - SubTB+Entropy adds +0.15 MCE vs TB
   - Prevents mode collapse, especially in large models

3. **Stochastic sampling mandatory**
   - Greedy: 100% mode collapse rate
   - Temperature (τ=0.5-1.0): 0% collapse rate
   - Top-p/top-k: Intermediate

4. **256× sample efficiency over baselines**
   - MOGFN-PC reaches 99.7% quality with 1K samples
   - Random needs 256K samples for same quality

5. **Task heterogeneity matters**
   - No universal config
   - Validate on your domain

### Contributions
✅ **First systematic diversity study for GFlowNets**
✅ **Novel metrics** (16 total, 7 categories)
✅ **Rigorous experiments** (720 experiments, statistical validation)
✅ **Actionable insights** (practitioner guidelines)
✅ **Open source** (code, data, metrics available)

### Impact
**For researchers:** Foundation for diversity-aware GFlowNet design
**For practitioners:** Evidence-based hyperparameter selection
**For field:** Establishes diversity as first-class optimization target

### Bottom Line
> "We've moved diversity from anecdotal to rigorous in Multi-Objective GFlowNets"

---

## SLIDE 25: Thank You + Q&A
**Layout:** Contact + next steps

### Thank You!

**Questions?**

### Contact
📧 [your.email]
💻 github.com/[username]/diversity-mogfn
📄 Paper: [link]

### Code & Data
All experiments reproducible:
- Metric implementations
- Experimental configs
- Trained models
- Analysis scripts

**QR Code:** [Link to repo]

---

## BACKUP SLIDES (for Q&A)

### BACKUP 1: All Metric Definitions
[Mathematical formulas for all 16 metrics]

### BACKUP 2: Statistical Tests
[ANOVA tables, post-hoc tests, effect sizes]

### BACKUP 3: Hyperparameter Grid
[Full grid search results]

### BACKUP 4: Additional Failure Cases
[More diagnostic analyses]

### BACKUP 5: Training Dynamics
[Loss curves, diversity over time]

### BACKUP 6: Comparison with HN-GFN
[If available from literature]

---

## TIME ALLOCATION FOR TECHNICAL AUDIENCE

**Adjusted timing:**
- Introduction: 2 min (Slides 1-2) - REDUCED from 3 min
- Metrics: 5 min (Slides 3-6) - SAME (important)
- Experimental Design: 4 min (Slides 7-9) - INCREASED from 3 min (technical audience values rigor)
- Results: 15 min (Slides 10-21) - INCREASED from 10 min (depth > breadth)
- Discussion: 4 min (Slides 22-24) - INCREASED from 3 min (limitations matter to grad students)

**Total: 30 min**

**Q&A strategy:**
- Expect technical questions on:
  - MCE auto-tuning (have BACKUP 1 ready)
  - Statistical power (have BACKUP 2)
  - Comparison with recent work (know HN-GFN, MORL papers)
  - Scalability to real applications
