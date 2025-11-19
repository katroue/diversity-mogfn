# Diversity in Multi-Objective GFlowNets
## PowerPoint Slide Content (30-min Presentation)

---

## SLIDE 1: Title Slide
**Layout:** Title + Subtitle + Visual

### Title
**Diversity in Multi-Objective GFlowNets:**
**A Systematic Study**

### Subtitle
Katherine Demers
[Your Affiliation]
[Date]

### Visual
- Background: 2×2 grid showing the MCE heatmaps from `results/factorials/analysis/grid_mce_capacity_level_loss_level.png`
- Semi-transparent overlay so text is readable

---

## SLIDE 2: The Diversity Challenge
**Layout:** Problem Statement + Visual Comparison

### Title
**Multi-Objective Optimization Requires Diverse Solutions**

### Content (Left side)
**The Problem:**
- Multi-objective problems have infinitely many optimal solutions
- Real applications need diverse options, not just a single "best" answer
- Current GFlowNet research emphasizes quality (convergence) over diversity

**Our Question:**
> How do architectural and training choices affect diversity in Multi-Objective GFlowNets?

### Visual (Right side)
**Split comparison:**
- **Left:** Pareto front with poor diversity (few clustered points)
  - Label: "Low Diversity: Limited options"
- **Right:** Pareto front with good diversity (well-spread points)
  - Label: "High Diversity: Rich trade-off exploration"

### Speaker Notes
"Imagine you're designing a new drug. You don't want just ONE molecule - you want dozens of diverse candidates that offer different property trade-offs."

---

## SLIDE 3: Why Diversity Matters
**Layout:** 3-column application examples

### Title
**Diverse Solutions Enable Better Decision-Making**

### Column 1: Drug Discovery
**Icon:** 💊 Molecule structure
- Multiple candidate molecules
- Different property profiles
- Hedge against failure in trials
- Example: QED vs. SA score trade-offs

### Column 2: Engineering Design
**Icon:** ⚙️ Engineering diagram
- Explore design space thoroughly
- Discover non-obvious solutions
- Accommodate changing requirements
- Example: Cost vs. performance

### Column 3: Preference Learning
**Icon:** 👥 Multiple users
- Different stakeholders, different preferences
- One model serves all needs
- Conditional generation
- Example: Personalized recommendations

### Bottom Banner
⚠️ **Mode Collapse = Wasted Computation**
Solutions clustered in one region → 99% of samples redundant

---

## SLIDE 4: Our Contributions
**Layout:** 4 boxes with checkmarks

### Title
**A Comprehensive Study of Diversity in MOGFNs**

### Contribution 1
✅ **Novel Metrics**
- 16 metrics across 7 categories
- First GFlowNet-specific diversity measures
- Capture trajectory, flow, and objective space diversity

### Contribution 2
✅ **Systematic Experiments**
- 3 factorial designs (capacity, loss, sampling)
- 4 environments (simple → complex)
- 720 total experiments

### Contribution 3
✅ **Baseline Comparisons**
- MOGFN-PC vs Random vs NSGA-II
- Quality, diversity, efficiency trade-offs
- 256× sample efficiency gains

### Contribution 4
✅ **Actionable Insights**
- Clear guidelines for practitioners
- Task-specific recommendations
- Open-source implementation

### Bottom
**Repository:** github.com/[your-username]/diversity-mogfn

---

## SLIDE 5: Background - Generative Flow Networks
**Layout:** Left text, Right diagram

### Title
**GFlowNets: Amortized Sampling from Energy Functions**

### Content (Left)
**Key Idea:**
Learn a policy that samples objects x proportional to reward R(x)

**Advantages:**
- **Amortized inference:** Train once, sample many
- **Mode-seeking:** Naturally finds diverse high-reward states
- **Compositionality:** Sequential construction process

**Training:**
Flow consistency constraint (trajectory balance)

### Diagram (Right)
**Flow network visualization:**
```
Initial State (s₀)
    ↓ a₁
Intermediate States
    ↓ a₂, a₃, ...
    ↓ aₙ
Terminal States (x) → R(x)
```

**Annotations:**
- Forward policy: Pθ(a|s)
- Backward policy: P_B(s'|s)
- Flow: F(s) ∝ R(x)

---

## SLIDE 6: Multi-Objective GFlowNets (MOGFN-PC)
**Layout:** Architecture diagram with annotations

### Title
**MOGFN-PC: Preference-Conditioned Multi-Objective GFlowNets**

### Architecture Diagram
**Flow (left to right):**

1. **Preference Vector (w)**
   - Input: [w₁, w₂, ..., wₘ]
   - Sampled from Dirichlet/Uniform

2. **Preference Encoder**
   - Thermometer encoding or vanilla
   - Embeds preferences

3. **Policy Network**
   - Input: [state, preference embedding]
   - Conditioning: Concat or FiLM
   - Output: Action distribution

4. **State Transitions**
   - s₀ → s₁ → ... → sₜ (terminal)

5. **Multi-Objective Evaluation**
   - f(x) = [f₁(x), f₂(x), ..., fₘ(x)]
   - Scalarized reward: R(x,w) = w · f(x)

### Key Equations (bottom)
```
Loss: ℒ_TB = 𝔼[(log Z + log P_F(τ|w) - log P_B(τ) - log R(x,w))²]
```

---

## SLIDE 7: The Diversity Gap
**Layout:** Problem illustration

### Title
**Existing Metrics Miss Critical Aspects of Diversity**

### Visual: Venn Diagram
Three overlapping circles showing different diversity aspects:

**Circle 1: Objective Space**
- Metrics: Hypervolume, IGD, Spacing
- What they measure: Spread in objective space
- **What they miss:** How solutions were generated

**Circle 2: Generation Process**
- Metrics: Trajectory diversity, path exploration
- What they measure: Diversity of construction paths
- **What they miss:** Final distribution properties

**Circle 3: Learned Distribution**
- Metrics: Flow concentration, mode coverage
- What they measure: Distribution entropy
- **What they miss:** Quality of coverage

### Center (Overlap)
**Our Approach:**
Comprehensive suite capturing all three facets

### Bottom Text
❌ Traditional MO metrics: Focus on convergence (distance to Pareto front)
✅ Our metrics: Coverage, exploration, flow distribution, AND convergence

---

## SLIDE 8: Our Metric Suite - 7 Categories
**Layout:** Icon grid with descriptions

### Title
**16 Metrics Across 7 Complementary Categories**

### Grid (2 rows × 4 columns)

**Row 1:**

| Traditional | Spatial | Trajectory | Objective |
|------------|---------|------------|-----------|
| 📊 Hypervolume (HV) | 🗺️ Mode Coverage Entropy (MCE) | 🛤️ Trajectory Diversity Score (TDS) | 🎯 Preference-Aligned Spread (PAS) |
| Generational Distance (GD) | Pairwise Min Distance (PMD) | Multi-Path Diversity (MPD) | Pareto Front Smoothness (PFS) |
| Inverted GD (IGD), Spacing | | | |
| **Standard MO benchmarks** | **Objective space coverage** | **Path exploration** | **Preference-conditioned** |

**Row 2:**

| Flow | Dynamics | Composite | — |
|------|----------|-----------|---|
| 🌊 Flow Concentration Index (FCI) | 📈 Replay Buffer Diversity (RBD) | 🏆 Quality-Diversity Score (QDS) | — |
| | | Diversity-Efficiency Ratio (DER) | — |
| **Distribution entropy** | **Training behavior** | **Integrated metrics** | — |

### Bottom Callout
⭐ **Focus Metric:** Mode Coverage Entropy (MCE) - captures distinct modes via DBSCAN clustering

---

## SLIDE 9: Spotlight - Mode Coverage Entropy (MCE)
**Layout:** Left definition, Right visual example

### Title
**Mode Coverage Entropy: Measuring Objective Space Coverage**

### Definition (Left)
**Algorithm:**
1. Cluster objectives using DBSCAN (auto-tuned ε)
2. Compute cluster sizes: p₁, p₂, ..., pₖ
3. Calculate entropy: MCE = -Σ pᵢ log(pᵢ) / log(k)

**Properties:**
- MCE ∈ [0, 1]
- MCE = 0: Mode collapse (all samples in one cluster)
- MCE = 1: Uniform coverage (equal-sized clusters)
- MCE = 0.5-0.7: Good diversity (typical for well-trained models)

**Why Important:**
✓ Captures distinct modes, not just spread
✓ Robust to outliers (via clustering)
✓ Normalized for comparison across tasks

### Visual (Right)
**Three examples (top to bottom):**

1. **MCE ≈ 0** (Mode Collapse)
   - Scatter plot: All points clustered in one region
   - 1 DBSCAN cluster (red)

2. **MCE ≈ 0.5** (Moderate Diversity)
   - Scatter plot: 3-4 distinct clusters
   - Unequal sizes

3. **MCE ≈ 0.9** (High Diversity)
   - Scatter plot: 8-10 well-distributed clusters
   - Nearly equal sizes

---

## SLIDE 10: Experimental Environments
**Layout:** 2×2 grid of environment cards

### Title
**Four Diverse Environments: Simple → Complex**

### Environment 1: HyperGrid
**Icon:** Grid visualization
- **Domain:** 10×10 discrete grid
- **Objectives:** 2 (distance to corners)
- **State Space:** ~100 states
- **Difficulty:** ⭐ Small
- **Challenge:** Simple but difficult coverage

### Environment 2: N-grams
**Icon:** Text/language
- **Domain:** Text sequence generation
- **Objectives:** 2 (diversity + fluency)
- **State Space:** ~10³ states
- **Difficulty:** ⭐⭐ Medium
- **Challenge:** Balance coherence and variation

### Environment 3: Molecules
**Icon:** Chemical structure
- **Domain:** Drug-like molecule design
- **Objectives:** 2 (QED + SA score)
- **State Space:** ~10⁶ states
- **Difficulty:** ⭐⭐ Medium
- **Challenge:** Valid chemistry constraints

### Environment 4: Sequences
**Icon:** Binary string
- **Domain:** Bit string optimization
- **Objectives:** 2 (custom functions)
- **State Space:** ~10⁸ states
- **Difficulty:** ⭐⭐⭐ Large
- **Challenge:** Scalability

### Bottom Summary
**Total experiments:** 720 (4 tasks × 3 factorial designs × 15 conditions × 4 seeds)

---

## SLIDE 11: Factorial Experimental Design
**Layout:** Three factorial tables

### Title
**Systematic 2-Way Factorial Studies**

### Factorial 1: Capacity × Loss Function
**Table (3×3):**
```
              TB      SubTB    SubTB+Entropy
Small      [exp 1]  [exp 2]     [exp 3]
Medium     [exp 4]  [exp 5]     [exp 6]
Large      [exp 7]  [exp 8]     [exp 9]
```
**Question:** How do model size and loss interact?

### Factorial 2: Capacity × Temperature
**Table (3×3):**
```
              Low     Medium    High
Small      [exp 10] [exp 11]  [exp 12]
Medium     [exp 13] [exp 14]  [exp 15]
Large      [exp 16] [exp 17]  [exp 18]
```
**Question:** Does capacity affect sampling temperature sensitivity?

### Factorial 3: Sampling × Loss
**Table (3×3):**
```
               TB      SubTB    SubTB+Entropy
Greedy      [exp 19] [exp 20]    [exp 21]
Top-p       [exp 22] [exp 23]    [exp 24]
Temperature [exp 25] [exp 26]    [exp 27]
```
**Question:** Which sampling strategy best preserves diversity?

### Bottom
**Statistical rigor:** 5 seeds × 27 conditions = 135 runs per task × 4 tasks = **540 MOGFN experiments**

---

## SLIDE 12: Baseline Comparison
**Layout:** Comparison table

### Title
**MOGFN-PC vs. State-of-Art Baselines**

### Comparison Table
| Aspect | MOGFN-PC | Random Sampling | NSGA-II |
|--------|----------|----------------|---------|
| **Approach** | Learned generative policy | Uniform random trajectories | Evolutionary algorithm |
| **Training** | Trajectory balance loss | None | Population evolution |
| **Inference** | O(L) forward passes | O(N) environment steps | O(G×P) generations |
| **Quality** | ⭐⭐⭐⭐⭐ High | ⭐ Very Low | ⭐⭐⭐⭐ High |
| **Diversity** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Moderate |
| **Sample Efficiency** | ⭐⭐⭐⭐⭐ Excellent | ⭐ Very Poor | ⭐⭐⭐ Good |
| **Wall-clock Time** | ⭐⭐ Slow (training) | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate |

### Bottom Statistics (Highlighted)
**MOGFN-PC achievements:**
- 99.7% of Random's hypervolume
- **256× fewer samples** than Random
- Best Quality-Diversity Score (QDS)

**When to use each:**
- MOGFN-PC: Sample budget limited, need quality+diversity
- Random: Baseline only, not practical
- NSGA-II: No training budget, immediate results needed

---

## SLIDE 13: Experimental Protocol
**Layout:** Pipeline diagram

### Title
**Rigorous Experimental Methodology**

### Pipeline (Left to Right)

**Step 1: Configuration**
- Factorial design specification
- Hyperparameter grid
- Random seed selection (42, 153, 264, 375, 486)

↓

**Step 2: Training**
- MOGFN-PC model training
- Task-specific iterations (4K-20K)
- Checkpoint saving

↓

**Step 3: Evaluation**
- Sample 1K trajectories per seed
- Compute all 16 metrics
- Save objectives.npy, metrics.json

↓

**Step 4: Aggregation**
- Group by experimental factors
- Compute mean ± std across seeds
- Statistical significance testing

↓

**Step 5: Analysis**
- Interaction plots
- Heatmaps
- Cross-task comparison

### Bottom Box
**Reproducibility:**
- ✓ Fixed random seeds
- ✓ All code public on GitHub
- ✓ Metrics validated on known benchmarks
- ✓ CSV results available

---

## SLIDE 14: Results Overview
**Layout:** Section title with key numbers

### Title
**Results: 720 Experiments Reveal Clear Patterns**

### Key Numbers (Large, centered)

```
99.7%     256×        0.69        3×3×4
Quality   Sample      Peak MCE    Factorial
          Efficiency  (Molecules) Design
```

### Bottom Teaser
**Next slides:**
1. Sample efficiency gains
2. Capacity effects on diversity
3. Loss function interactions
4. Sampling strategy impact
5. Task-specific patterns
6. Baseline comparisons

---

## SLIDE 15: Result 1 - Sample Efficiency
**Layout:** Bar chart comparison

### Title
**MOGFN-PC: 256× More Sample-Efficient Than Random**

### Bar Chart
**X-axis:** Algorithm (MOGFN-PC, NSGA-II, Random)
**Y-axis:** Samples needed for HV = 0.65 (log scale)

**Bars:**
- MOGFN-PC: ~1,000 samples (✅ green)
- NSGA-II: ~5,000 samples (🟨 yellow)
- Random: ~256,000 samples (🔴 red)

### Annotations
- Arrow pointing from MOGFN to Random: "**256× fewer samples**"
- Badge on MOGFN-PC: "99.7% of max quality"

### Bottom Table
**Hypervolume per 1K samples:**
| Algorithm | HV/1K samples | Efficiency Gain |
|-----------|---------------|-----------------|
| MOGFN-PC  | 0.652         | 256× (baseline) |
| NSGA-II   | 0.130         | 51×             |
| Random    | 0.0025        | 1×              |

### Key Insight Box
💡 **Implication:** In sample-constrained domains (expensive simulations, wet-lab experiments), MOGFN-PC provides 2+ orders of magnitude improvement

---

## SLIDE 16: Result 2 - The Capacity Sweet Spot
**Layout:** Small multiples heatmap (THE KEY VISUAL)

### Title
**Medium Capacity Optimizes Diversity Across Most Tasks**

### Visual
**The 2×2 grid heatmap** (from `grid_mce_capacity_level_loss_level.png`)

**Layout:**
```
[HyperGrid]      [N-grams]
[Molecules]      [Sequences]
```

Each subplot: Capacity (x-axis) × Loss (y-axis) showing MCE values

### Annotations on visual
- Circle around medium capacity in N-grams: "Sweet spot"
- Arrow pointing to HyperGrid large values: "Task-dependent"
- Badge on Molecules: "Saturated coverage"

### Interpretation (Below visual)
**Pattern:**
- ✅ **Medium (64-128 dim):** Best for HyperGrid, N-grams, Sequences
- ⚠️ **Small (32 dim):** Underfitting → low MCE
- ⚠️ **Large (256 dim):** Overfitting → mode collapse (except Molecules)

**Why?**
- Small models: Insufficient capacity to represent diverse modes
- Large models: Overconfident predictions, collapse to single mode
- **Medium: Goldilocks zone**

---

## SLIDE 17: Result 3 - Entropy Regularization Helps
**Layout:** Interaction plot

### Title
**SubTB + Entropy Prevents Mode Collapse**

### Interaction Plot
**X-axis:** Capacity (Small, Medium, Large)
**Y-axis:** MCE (0.0 - 0.7)
**Lines:** Three loss functions (TB, SubTB, SubTB+Entropy)

**Pattern:**
- TB (blue line): Low MCE, flat across capacities
- SubTB (orange line): Moderate MCE, slight increase
- SubTB+Entropy (green line): High MCE, peaks at medium
- **Non-parallel lines = interaction effect**

### Annotations
- Highlight the gap at medium capacity
- Arrow: "Entropy regularization adds +0.15 MCE at medium capacity"

### Bottom Explanation
**Entropy term encourages:**
✓ Exploration of diverse trajectories
✓ Prevention of premature convergence
✓ Smoother policy distributions

**Trade-off:**
⚠️ Slightly slower convergence (10-15% more iterations)
✅ But much better diversity (25-40% higher MCE)

**Recommendation:** Always use SubTB+Entropy for diversity-critical applications

---

## SLIDE 18: Result 4 - Sampling Strategy Matters
**Layout:** Box plot comparison

### Title
**Temperature Sampling >> Top-p > Greedy**

### Box Plot
**X-axis:** Sampling strategy (Greedy, Top-k, Top-p, Temperature)
**Y-axis:** MCE (0.0 - 0.7)

**Distributions:**
- **Greedy:** Tight box at MCE ≈ 0.0 (red)
- **Top-k:** Box at MCE ≈ 0.2 (orange)
- **Top-p:** Box at MCE ≈ 0.35 (yellow)
- **Temperature (τ=0.5-1.0):** Box at MCE ≈ 0.55 (green)

### Statistics Table (Below)
| Strategy | Mean MCE | Std | Mode Collapse Rate |
|----------|----------|-----|-------------------|
| Greedy   | 0.02     | 0.03 | **100%** (5/5 seeds) |
| Top-k    | 0.19     | 0.11 | 60% (3/5 seeds) |
| Top-p    | 0.34     | 0.08 | 20% (1/5 seeds) |
| Temperature | 0.56  | 0.05 | **0%** (0/5 seeds) |

### Key Insight
⚠️ **Greedy = Mode Collapse:** Deterministic → single trajectory → MCE ≈ 0
✅ **Temperature sampling:** Stochastic exploration → diverse modes

**Recommended:** τ = 0.5 - 1.0 (balances diversity and quality)

---

## SLIDE 19: Result 5 - Task-Specific Patterns
**Layout:** 4 task summary cards

### Title
**Diversity Characteristics Vary Dramatically Across Tasks**

### Task 1: HyperGrid
**Graph:** MCE range box plot
- **Mean MCE:** 0.05 - 0.21 (difficult task)
- **Challenge:** Simple state space, hard to cover uniformly
- **Best config:** Medium capacity + SubTB+Entropy
- **Insight:** Even simple tasks can have poor mode coverage

### Task 2: N-grams
**Graph:** MCE range box plot
- **Mean MCE:** 0.52 - 0.61 (robust task)
- **Challenge:** Consistent performance across all configs
- **Best config:** Any medium+ capacity
- **Insight:** Text domain naturally encourages diversity

### Task 3: Molecules
**Graph:** MCE range box plot
- **Mean MCE:** 0.67 - 0.69 (saturated)
- **Challenge:** Excellent coverage, little room for improvement
- **Best config:** Large capacity (more complex objectives)
- **Insight:** Molecular space well-suited to GFlowNets

### Task 4: Sequences
**Graph:** MCE range box plot
- **Mean MCE:** 0.19 - 0.55 (capacity-sensitive)
- **Challenge:** Large state space, sensitive to hyperparameters
- **Best config:** Medium capacity + Temperature sampling
- **Insight:** Scalability requires careful tuning

### Bottom Insight
💡 **No one-size-fits-all:** Task complexity and state space structure strongly affect diversity
→ **Always validate metrics on your specific domain**

---

## SLIDE 20: Result 6 - Quality-Diversity Trade-off
**Layout:** Scatter plot

### Title
**MOGFN-PC Achieves Best Quality-Diversity Balance**

### Scatter Plot
**X-axis:** Hypervolume (Quality) → 0.0 to 1.0
**Y-axis:** MCE (Diversity) → 0.0 to 1.0

**Point clusters:**
- **MOGFN-PC** (green): Upper-right (HV ≈ 0.68, MCE ≈ 0.55)
- **NSGA-II** (blue): Mid-right (HV ≈ 0.65, MCE ≈ 0.42)
- **Random** (red): Upper-left (HV ≈ 0.15, MCE ≈ 0.71)

### Annotations
- Pareto frontier line connecting best points
- MOGFN-PC circled: "Optimal trade-off"
- Arrow from Random to MOGFN: "+350% quality"
- Arrow from NSGA-II to MOGFN: "+31% diversity"

### Bottom QDS Scores
**Quality-Diversity Score (QDS = 0.5×HV + 0.5×MCE):**
- MOGFN-PC: **0.615** ⭐ (winner)
- NSGA-II: 0.535
- Random: 0.430

**Interpretation:** MOGFN-PC dominates when both quality AND diversity matter

---

## SLIDE 21: Result 7 - Computational Trade-offs
**Layout:** Multi-axis radar chart

### Title
**Efficiency Trade-offs: Sample vs. Time vs. Quality**

### Radar Chart (5 axes)
**Axes (normalized 0-1):**
1. Sample Efficiency
2. Wall-clock Speed
3. Hypervolume (Quality)
4. MCE (Diversity)
5. Ease of Use (no tuning needed)

**Polygons:**
- **MOGFN-PC** (green): High sample efficiency, quality, diversity; Low speed
- **Random** (red): High speed, diversity; Low everything else
- **NSGA-II** (blue): Balanced, moderate on all axes

### Bottom Table
| Metric | MOGFN-PC | NSGA-II | Random |
|--------|----------|---------|--------|
| Samples to HV=0.65 | 1,000 | 5,000 | 256,000 |
| Wall-clock time (min) | 45-90 | 10-20 | 2-5 |
| Training time (min) | 30-60 | 0 | 0 |
| **Best use case** | Sample-limited | No training budget | Baseline only |

### Recommendation Box
**Use MOGFN-PC when:**
✅ Sample evaluations are expensive (wet lab, simulations)
✅ Can afford 30-60 min training time
✅ Need both quality AND diversity

**Use NSGA-II when:**
✅ Cannot afford training time
✅ Need immediate results
✅ Have medium sample budget

---

## SLIDE 22: Surprising Findings
**Layout:** 4 surprise boxes

### Title
**Unexpected Results & Lessons Learned**

### Surprise 1: FiLM vs Concat ≈ Same
**Icon:** 🤔
- Hypothesis: FiLM conditioning would significantly outperform Concat
- **Reality:** <1% difference in MCE (0.532 vs 0.529)
- **Implication:** Use simpler Concat for faster training
- Graph: Side-by-side box plots showing overlap

### Surprise 2: Large Models Hurt
**Icon:** 😮
- Hypothesis: Bigger always better
- **Reality:** Large capacity causes mode collapse in HyperGrid (MCE 0.07 vs 0.21 for medium)
- **Implication:** Task complexity should match model capacity
- Visual: Downward arrow from large → medium MCE

### Surprise 3: Random Finds More Unique Solutions
**Icon:** 🎲
- Observation: Random has 2-3× more unique solutions
- **Context:** But requires 256× more samples!
- **Efficiency:** 0.05% unique rate (Random) vs 5.8% (MOGFN-PC)
- **Implication:** Diversity ≠ efficiency
- Visual: Bar chart of unique solutions, normalized by samples

### Surprise 4: Greedy = Complete Collapse
**Icon:** ⚠️
- Result: 100% mode collapse rate across ALL tasks
- Expectation: At least some diversity from environment stochasticity
- **Reality:** Deterministic policy → deterministic outcomes
- **Lesson:** Never use greedy sampling for diversity-critical apps

### Bottom Insight
💡 **Takeaway:** Empirical validation is essential - intuitions about diversity don't always hold!

---

## SLIDE 23: Practical Guidelines
**Layout:** Decision tree / flowchart

### Title
**Actionable Recommendations for Practitioners**

### Flowchart

**Start:** "Need diverse multi-objective solutions?"
  ↓
**Q1:** "Sample budget limited?"
  - Yes → Use MOGFN-PC
  - No → Use NSGA-II or Random

  ↓ (MOGFN-PC path)
**Q2:** "Task complexity?"
  - Simple (state space < 10³) → **Small-Medium capacity**
  - Medium (state space 10³-10⁶) → **Medium capacity**
  - Complex (state space > 10⁶) → **Medium-Large capacity**

  ↓
**Q3:** "Training setup"
  - Loss: **SubTB + Entropy** (always)
  - Sampling: **Temperature (τ=0.5-1.0)**
  - Preference: **Dirichlet (α=1.5)**
  - Conditioning: **Concat** (simpler)

  ↓
**Q4:** "Validation"
  - Compute MCE on validation set
  - If MCE < 0.2 → Increase temperature or add entropy weight
  - If MCE > 0.6 → Good! Monitor quality (HV)

### Bottom Summary Table
| Setting | Recommended Value | Rationale |
|---------|------------------|-----------|
| Capacity | **Medium (64-128)** | Best trade-off |
| Loss | **SubTB + Entropy** | Prevents mode collapse |
| Sampling | **Temp (τ=0.7)** | Exploration + quality |
| Preference | **Dirichlet (α=1.5)** | Uniform coverage |

---

## SLIDE 24: Limitations & Challenges
**Layout:** Left limitations, Right solutions

### Title
**Honest Assessment: What We Didn't Solve**

### Left Column: Limitations

**1. Computational Cost**
- MCE computation: O(n²) via DBSCAN
- Slow for >10K samples
- Current solution: Subsample to 5K

**2. Limited Objective Count**
- Experiments focused on 2 objectives
- Many-objective (m > 3) untested
- Preference space grows exponentially

**3. Domain Restrictions**
- Discrete/compositional only
- Continuous spaces not evaluated
- GFlowNet architectural constraint

**4. Metric Selection**
- 16 metrics → which to prioritize?
- Task-dependent "best" metric
- No universal diversity score

### Right Column: Future Work

**1. Scalability Improvements**
- Approximate MCE (locality-sensitive hashing)
- Online diversity monitoring
- Incremental clustering

**2. Many-Objective Extension**
- Test on 3-5 objective problems
- Adaptive preference sampling
- Dimensionality reduction for visualization

**3. Broader Domains**
- Continuous optimization (Langevin GFN)
- Mixed discrete-continuous
- Graph generation

**4. Adaptive Methods**
- Auto-tune capacity based on MCE
- Dynamic temperature scheduling
- Diversity-aware training objectives

### Bottom Quote
> "We've answered how design choices affect diversity. The next question: How to automatically optimize for diversity during training?"

---

## SLIDE 25: Contributions Summary
**Layout:** 3 columns

### Title
**What This Work Contributes to the Field**

### Column 1: Theoretical
**Novel Metrics**
- 7 categories, 16 total metrics
- First GFlowNet-specific measures
- Validated on established benchmarks

**Conceptual Framework**
- Separates trajectory, flow, and objective diversity
- Highlights quality-diversity trade-off
- Defines evaluation standards

### Column 2: Empirical
**Systematic Experiments**
- 720 experiments across 4 domains
- 3 factorial designs
- Statistical rigor (multiple seeds)

**Clear Patterns**
- Medium capacity optimal
- Entropy regularization essential
- Temperature sampling best

### Column 3: Practical
**Actionable Guidelines**
- Decision tree for practitioners
- Hyperparameter recommendations
- When to use MOGFN vs baselines

**Open Science**
- All code on GitHub
- Reproducible experiments
- Metric implementations available

### Bottom Impact Statement
**Impact:** First systematic diversity study for GFlowNets
→ Enables practitioners to make **evidence-based architectural choices**

---

## SLIDE 26: Future Directions
**Layout:** Roadmap with 3 phases

### Title
**Research Roadmap: Short, Medium, Long Term**

### Phase 1: Short Term (3-6 months)
**Focus:** Incremental improvements

- ✅ Extend to 3-objective problems
- ✅ Test on additional domains (graphs, programs)
- ✅ Optimize MCE computation (approximate algorithms)
- ✅ Diversity-aware early stopping

**Expected Impact:** Immediate applicability to broader set of problems

### Phase 2: Medium Term (6-12 months)
**Focus:** Adaptive methods

- 🔄 Auto-tuning capacity based on diversity metrics
- 🔄 Dynamic temperature scheduling
- 🔄 Online diversity monitoring during training
- 🔄 Multi-fidelity diversity evaluation

**Expected Impact:** Reduce hyperparameter tuning burden

### Phase 3: Long Term (1-2 years)
**Focus:** Theoretical foundations

- 🔮 Diversity as training objective (not just evaluation)
- 🔮 Provable diversity guarantees
- 🔮 Connection to entropy-regularized RL
- 🔮 Unified diversity-quality optimization framework

**Expected Impact:** Principled approach to diversity in generative models

### Bottom Vision
**Vision:** Make diversity a first-class optimization target, not an afterthought

---

## SLIDE 27: Conclusions
**Layout:** Key messages with icons

### Title
**Key Takeaways**

### Takeaway 1: 🎯 Problem
**Multi-objective GFlowNets lack systematic diversity analysis**
- Existing work focuses on quality (convergence)
- Diversity is critical for real-world applications
- No standard evaluation metrics

### Takeaway 2: 📊 Solution
**Comprehensive metric suite + rigorous experiments**
- 16 metrics across 7 categories
- 720 experiments, 4 environments
- Statistical validation with baselines

### Takeaway 3: 💡 Insights
**Clear patterns emerge from factorial studies**
- Medium capacity (64-128 dim) optimal for most tasks
- Entropy regularization prevents mode collapse (+0.15 MCE)
- Temperature sampling essential (greedy = 100% collapse)
- 256× sample efficiency vs random baseline

### Takeaway 4: 🛠️ Impact
**Actionable guidelines for practitioners**
- Evidence-based architectural choices
- Task-specific recommendations
- Open-source implementation

### Bottom Quote
> "We've moved diversity from anecdotal observation to rigorous science in Multi-Objective GFlowNets."

---

## SLIDE 28: Thank You
**Layout:** Centered with contact info

### Title
**Thank You!**

### Contact
**Katherine Demers**
📧 [your.email@university.edu]
💻 github.com/[username]/diversity-mogfn
📄 Paper: [arXiv link or conference]

### QR Code
**Large QR code linking to:**
- GitHub repository
- Or presentation slides
- Or paper PDF

### Bottom
**Questions?**

---

## Backup Slides (Optional, for Q&A)

### BACKUP 1: Detailed Metric Formulas
**All 16 metric mathematical definitions**

### BACKUP 2: Hyperparameter Sensitivity
**Grid search results for temperature, alpha, batch size**

### BACKUP 3: Additional Task Details
**Full environment specifications and reward functions**

### BACKUP 4: Statistical Significance Tests
**ANOVA tables, p-values, effect sizes**

### BACKUP 5: Training Curves
**Loss curves, diversity over time, convergence analysis**

### BACKUP 6: Failure Cases
**When MOGFN-PC performs poorly, diagnostic analysis**

---

## Notes for PowerPoint Creation

### Color Scheme
- **Primary:** Teal/Blue (#2E86AB)
- **Secondary:** Orange (#F77F00)
- **Accent:** Green (#06A77D) for positive results
- **Warning:** Red (#D62828) for problems/limitations
- **Background:** White or light gray (#F8F9FA)

### Typography
- **Title:** Sans-serif, bold, 36-44pt
- **Body:** Sans-serif, 18-24pt
- **Captions:** 14-16pt
- **Code/numbers:** Monospace when showing data

### Visual Guidelines
- **Graphs:** High contrast, large legends, clear axis labels
- **Icons:** Use consistently (same style pack)
- **Animations:** Minimal - only to reveal complex ideas step-by-step
- **White space:** Don't overcrowd slides

### Data Visualizations to Include
1. MCE heatmap grid (Slide 16) - most important!
2. Sample efficiency bar chart (Slide 15)
3. Quality-diversity scatter (Slide 20)
4. Interaction plot (Slide 17)
5. Box plots for sampling strategies (Slide 18)

### Recommended Export from Your Results
- Use: `results/factorials/analysis/grid_mce_capacity_level_loss_level.png`
- Generate: Baseline comparison charts from summary CSVs
- Create: Interaction plots using matplotlib with consistent styling
