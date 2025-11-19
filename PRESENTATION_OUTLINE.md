# Diversity in Multi-Objective GFlowNets: A Systematic Study
## 30-Minute Presentation Outline

**Presenter:** [Your Name]
**Duration:** 30 minutes (25 min talk + 5 min Q&A)
**Audience:** Academic/Research audience familiar with ML basics

---

## Slide Breakdown (Total: ~27 slides)

### Section 1: Introduction & Motivation (3 minutes, Slides 1-4)

**Slide 1: Title Slide**
- Title: "Diversity in Multi-Objective GFlowNets: A Systematic Study"
- Your name, affiliation, date
- One compelling visualization (e.g., the 2×2 grid of MCE heatmaps)

**Slide 2: The Challenge**
- **Problem:** Multi-objective optimization requires diverse, high-quality solutions
- **Gap:** Existing GFlowNet research focuses on quality (Pareto optimality) but lacks systematic diversity analysis
- **Question:** How do architectural and training choices affect diversity in MOGFNs?
- Visual: Side-by-side comparison of low diversity vs high diversity Pareto fronts

**Slide 3: Why Diversity Matters**
- Real-world applications need options, not just optimal points
- Drug discovery: Multiple candidate molecules with different properties
- Engineering design: Trade-off exploration requires coverage
- Decision making: Stakeholders have varying preferences
- Visual: Application examples with diverse vs collapsed solutions

**Slide 4: Our Contributions**
- ✓ Novel GFlowNet-specific diversity metrics (16 metrics, 7 categories)
- ✓ Comprehensive factorial studies across 4 environments
- ✓ Systematic baseline comparisons (MOGFN-PC, Random, NSGA-II)
- ✓ Actionable insights for practitioners
- Visual: Project overview diagram

---

### Section 2: Background (5 minutes, Slides 5-9)

**Slide 5: Generative Flow Networks (GFlowNets)**
- Sequential decision-making framework
- Samples proportional to rewards R(x)
- Key advantage: Amortized inference (one model → many samples)
- Diagram: GFlowNet generation process (states → actions → terminal states)

**Slide 6: Multi-Objective GFlowNets (MOGFN-PC)**
- Extension: Multiple conflicting objectives
- Preference-conditioned generation
- Architecture: Policy network + backward policy + preference encoder
- Diagram: MOGFN-PC architecture with preference conditioning

**Slide 7: The Diversity Problem**
- Traditional MO metrics focus on convergence (how close to Pareto front?)
- But what about spread, coverage, exploration?
- GFlowNets learn distributions - need flow-specific metrics
- Visual: Examples of high convergence but low diversity

**Slide 8: Existing Metrics are Insufficient**
- Traditional: Hypervolume, IGD (convergence-focused)
- Miss: Trajectory diversity, flow concentration, mode coverage
- Need: Comprehensive suite capturing multiple facets
- Table: Comparison of metric categories

**Slide 9: Our Approach**
- Systematic factorial experiments
- Multiple environments (simple → complex)
- Comprehensive metric suite
- Statistical rigor (5 seeds per condition)
- Diagram: Experimental pipeline

---

### Section 3: Methodology (7 minutes, Slides 10-16)

**Slide 10: Diversity Metrics - 7 Categories**
- **Traditional**: Hypervolume, GD, IGD, Spacing
- **Spatial**: Mode Coverage Entropy (MCE), Pairwise Min Distance (PMD)
- **Trajectory**: Trajectory Diversity Score (TDS), Multi-Path Diversity (MPD)
- **Objective**: Preference-Aligned Spread (PAS), Pareto Front Smoothness (PFS)
- **Flow**: Flow Concentration Index (FCI)
- **Dynamics**: Replay Buffer Diversity (RBD)
- **Composite**: Quality-Diversity Score (QDS), Diversity-Efficiency Ratio (DER)
- Visual: Icon/diagram for each category

**Slide 11: Key Metric: Mode Coverage Entropy (MCE)**
- Measures objective space coverage using DBSCAN clustering
- MCE = H(p₁, p₂, ..., pₖ) where pᵢ = cluster size / total
- MCE ∈ [0, 1]: 0 = mode collapse, 1 = uniform coverage
- Why important: Captures distinct modes, not just spread
- Visual: Examples of low MCE vs high MCE with clusters

**Slide 12: Experimental Environments**
- **HyperGrid** (small): 10×10 grid, 2 objectives
- **N-grams** (medium): Text generation, diversity + fluency
- **Molecules** (medium): Drug design, QED + SA score
- **Sequences** (large): Bit strings, multi-objective optimization
- Visual: Grid showing environment characteristics (state space, objective space)

**Slide 13: Factorial Design**
1. **Capacity × Loss** (3×3 design, 45 experiments per task)
   - Capacity: small/medium/large
   - Loss: TB / SubTB / SubTB+Entropy

2. **Capacity × Sampling** (3×3 design)
   - Capacity: small/medium/large
   - Temperature: low/medium/high

3. **Sampling × Loss** (3×3 design)
   - Sampling: greedy/top-p/temperature
   - Loss: TB / SubTB / SubTB+Entropy

**Slide 14: Baseline Algorithms**
- **MOGFN-PC**: Our method (preference-conditioned GFlowNet)
- **Random Sampling**: Naive baseline
- **NSGA-II**: State-of-art evolutionary algorithm (pymoo)
- Comparison axes: Quality, diversity, sample efficiency, wall-clock time
- Visual: Algorithm comparison table

**Slide 15: Experimental Protocol**
- 5 random seeds per condition (statistical rigor)
- Task-specific iteration budgets (validated in ablations)
- All 16 metrics computed per experiment
- Results aggregated: mean ± std across seeds
- Infrastructure: PyTorch, scikit-learn, pymoo

**Slide 16: Analysis Methods**
- Factorial interaction plots (capacity × loss effects)
- Small multiples heatmaps (cross-task comparison)
- Statistical significance testing
- Visual: Example interaction plot

---

### Section 4: Results (10 minutes, Slides 17-24)

**Slide 17: Key Finding 1 - Sample Efficiency**
- **MOGFN-PC achieves 99.7% of Random's quality with 0.8% of samples**
- Hypervolume comparison (per 1K samples)
- MOGFN-PC: 130-256× more sample-efficient than Random
- Visual: Bar chart comparing sample efficiency

**Slide 18: Key Finding 2 - Capacity Effects on Diversity**
- **Medium capacity optimal for most tasks**
- Small: Underfitting → low coverage
- Large: Overfitting → mode collapse
- Task-dependent: Molecules prefer large, HyperGrid prefers medium
- Visual: The 2×2 MCE heatmap grid (capacity × loss across tasks)

**Slide 19: Capacity × Loss Interaction**
- **SubTB + Entropy regularization improves diversity**
- Entropy term prevents mode collapse
- Effect strongest for small/medium capacity
- Large capacity: Loss function matters less
- Visual: Interaction plot showing non-parallel lines

**Slide 20: Key Finding 3 - Sampling Strategy Impact**
- **Temperature sampling > Top-p > Greedy**
- Greedy: Mode collapse (MCE ≈ 0) across all seeds
- Low temperature: Deterministic → poor diversity
- High temperature: Better exploration but noisy
- Visual: Box plots comparing sampling strategies

**Slide 21: Task-Specific Patterns**
- **HyperGrid**: Low MCE (0.0-0.2), difficult coverage
- **N-grams**: Consistent MCE (0.5-0.6), robust to settings
- **Molecules**: Excellent MCE (≈0.69), saturated coverage
- **Sequences**: Variable MCE (0.2-0.5), capacity-sensitive
- Visual: Small multiples showing task differences

**Slide 22: Baseline Comparison - Quality vs Diversity**
- MOGFN-PC: Best quality-diversity balance (QDS)
- Random: High diversity, poor quality
- NSGA-II: Good quality, moderate diversity
- Visual: Scatter plot (Hypervolume vs MCE) with algorithm clusters

**Slide 23: Baseline Comparison - Efficiency Trade-offs**
- Sample efficiency: MOGFN-PC >> NSGA-II >> Random
- Wall-clock time: Random < NSGA-II < MOGFN-PC
- Training overhead: MOGFN-PC 35-90× slower than Random
- **Recommendation**: Use MOGFN-PC when sample budget is limited
- Visual: Multi-axis comparison chart

**Slide 24: Surprising Findings**
- ✓ FiLM vs Concat conditioning: <1% difference (unexpected!)
- ✓ Large models don't always help (overfitting in HyperGrid)
- ✓ Entropy regularization crucial for small models
- ✓ Random sampling finds more unique solutions but with 256× more evaluations
- Visual: Highlighted surprising result graphs

---

### Section 5: Conclusions & Future Work (3 minutes, Slides 25-27)

**Slide 25: Practical Guidelines**
- **Model Capacity**: Start with medium, tune based on task complexity
- **Loss Function**: Use SubTB + Entropy for better diversity
- **Sampling**: Temperature sampling with τ=0.5-1.0
- **Preference Distribution**: Dirichlet (α=1.5) for uniform coverage
- **When to use MOGFN-PC**: Sample-constrained, need quality-diversity balance
- Visual: Decision tree or flowchart

**Slide 26: Limitations & Future Work**
- **Limitations:**
  - Computational cost (DBSCAN O(n²) for MCE)
  - 2-objective focus (need to test on many-objective problems)
  - Limited to discrete/compositional domains

- **Future Work:**
  - Scale to 3+ objectives
  - Online diversity monitoring during training
  - Adaptive capacity/temperature based on diversity metrics
  - Integration with preference learning

**Slide 27: Summary**
- **Problem:** Systematic diversity analysis missing in MOGFN literature
- **Solution:** Comprehensive metric suite + factorial experiments
- **Key Insights:** Medium capacity, entropy regularization, temperature sampling
- **Impact:** First rigorous diversity study for GFlowNets
- **Reproducibility:** All code, data, and metrics available
- Visual: Project summary infographic

**Slide 28: Thank You + Q&A**
- Thank you message
- Contact information
- GitHub repository link
- QR code to repo/paper
- "Questions?"

---

## Speaker Notes & Timing Tips

### Timing Breakdown:
- **Introduction (Slides 1-4):** 3 minutes (~45 sec/slide)
- **Background (Slides 5-9):** 5 minutes (~1 min/slide)
- **Methodology (Slides 10-16):** 7 minutes (~1 min/slide)
- **Results (Slides 17-24):** 10 minutes (~1.25 min/slide) - MOST TIME HERE
- **Conclusions (Slides 25-27):** 3 minutes (~1 min/slide)
- **Q&A:** 2 minutes buffer

### Presentation Tips:

1. **Start Strong (Slide 2-3)**
   - Open with concrete example of why diversity matters
   - Make it relatable before diving into technical details

2. **Methodology Section (Slides 10-16)**
   - Don't read metric definitions - explain intuition
   - Focus on MCE as the "star metric" (Slide 11)
   - Briefly mention others, refer to paper for details

3. **Results Section (Slides 17-24)**
   - This is your meat - spend 40% of talk time here
   - Lead with sample efficiency (impressive number: 256×!)
   - Use the small multiples grid (Slide 18) as anchor visual
   - Tell a story: capacity → loss → sampling → baselines

4. **Visual Emphasis:**
   - Slide 18 (MCE grid): "This single figure summarizes 720 experiments"
   - Slide 17 (sample efficiency): "99.7% quality with 0.8% samples"
   - Slide 19 (interaction): "Non-parallel lines show true interaction"

5. **Anticipated Questions:**
   - "Why not use existing MO metrics?" → Slide 8 (they miss key aspects)
   - "How does this compare to MORL?" → Different paradigm (no critic needed)
   - "Computational cost?" → Slide 23 (35-90× slower wall-clock)
   - "Which metric is most important?" → MCE for coverage, QDS for overall

6. **Backup Slides (Optional, don't count in 30 min):**
   - Detailed metric formulas
   - Additional environment details
   - Full factorial tables
   - Hyperparameter sensitivity
   - Error analysis

---

## Key Messages to Emphasize

1. **Novel Contribution:** First systematic diversity study for GFlowNets
2. **Comprehensive:** 16 metrics × 4 environments × 3 factorial designs
3. **Practical Impact:** Clear guidelines for practitioners
4. **Rigorous:** Statistical testing, multiple seeds, baseline comparisons
5. **Open Science:** All code, data, and metrics publicly available

---

## Rehearsal Checklist

- [ ] Practice full talk under 25 minutes (leave 5 min for Q&A)
- [ ] Memorize opening hook (Slide 2)
- [ ] Know Slide 18 (MCE grid) cold - you'll refer to it multiple times
- [ ] Prepare 1-sentence summary of each metric category
- [ ] Have concrete numbers ready: 256× efficiency, 99.7% quality, 720 experiments
- [ ] Practice transitions between sections
- [ ] Prepare for "so what?" question - emphasize Slide 25 (practical guidelines)

---

## Visual Design Notes

For PowerPoint creation:
- **Color scheme:** Professional (blue/teal for MOGFN-PC, orange for baselines)
- **Consistent layout:** Title + content, avoid clutter
- **Font:** Sans-serif, minimum 18pt for body text
- **Graphs:** High contrast, large legends, clear axis labels
- **Animations:** Minimal - only to reveal complex ideas step-by-step
- **Key visuals:**
  - MCE grid heatmap (Slide 18) - largest, most important
  - Sample efficiency bar chart (Slide 17) - make numbers pop
  - Architecture diagram (Slide 6) - clear, not too detailed
  - Interaction plots (Slide 19) - annotate the non-parallel lines
