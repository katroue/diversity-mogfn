# Where Are FCI and PFS Important Metrics?

## Quick Answer

**FCI (Flow Concentration Index)** and **PFS (Pareto Front Smoothness)** are important in:

1. **Loss Ablation Study** (`configs/ablations/loss_ablation_final.yaml`)
2. **All 2-Way Factorial Studies** (capacity × sampling, capacity × loss, sampling × loss)

---

## Flow Concentration Index (FCI)

### Definition
**Flow Concentration Index (FCI)** measures how concentrated or dispersed the flow probabilities are across the action space in GFlowNets.

### Why It's Important

#### 1. **Loss Ablation Study** ⭐ PRIMARY USE CASE

**File:** `configs/ablations/loss_ablation_final.yaml`

**Why FCI matters here:**
- Different **loss functions** (TB, DB, SubTB, FM) affect how flows are distributed
- **Entropy regularization** explicitly aims to spread flows more evenly
- FCI directly measures the impact of loss modifications on flow distribution

**Key hypotheses:**
```yaml
h2: Entropy regularization significantly increases diversity
    prediction: "entropy(β=0.05) increases MCE by 20-30%"
    → FCI measures if flows become less concentrated (more diverse)

h4: SubTB(0.9) + entropy(β=0.05) is optimal combination
    → FCI validates that this combination spreads flows appropriately
```

**Expected insights:**
- **Trajectory Balance (TB)**: Moderate FCI (balanced flow concentration)
- **Detailed Balance (DB)**: High FCI (concentrated flows on greedy paths)
- **Flow Matching (FM)**: Low FCI (dispersed flows, explores broadly)
- **Entropy regularization**: Lower FCI (forces exploration, spreads flows)
- **SubTrajectory Balance (SubTB)**: Interpolates between TB and DB based on λ

**Why this is critical:**
FCI reveals whether a loss function encourages **exploration** (low FCI) or **exploitation** (high FCI), which directly impacts diversity.

---

#### 2. **Factorial Studies** (All 2-Way Interactions)

**Files:**
- `configs/factorials/capacity_sampling_2way.yaml`
- `configs/factorials/capacity_loss_2way.yaml`
- `configs/factorials/sampling_loss_2way.yaml`

**Why FCI matters in factorials:**

##### **Capacity × Sampling Factorial:**
```yaml
Question: Does capacity affect how temperature spreads flows?
FCI insight:
  - High capacity + low temperature → High FCI (overfitting to mode)
  - Low capacity + high temperature → Low FCI (underfitting, random flows)
```

##### **Capacity × Loss Factorial:**
```yaml
Question: Do larger models need different loss functions?
FCI insight:
  - Large model + TB → May have concentrated flows (mode collapse risk)
  - Large model + SubTB → Better flow distribution (prevents collapse)
```

##### **Sampling × Loss Factorial:**
```yaml
Question: How do sampling strategies interact with loss functions?
FCI insight:
  - Greedy sampling + TB → Very high FCI (deterministic, concentrated)
  - Categorical sampling + entropy loss → Low FCI (stochastic, dispersed)
```

**Factorial-specific insight:**
FCI helps detect **interactions** where certain combinations (e.g., small capacity + entropy loss) might cause flows to become too dispersed (poor quality) or too concentrated (poor diversity).

---

### What FCI Measures Technically

**Formula (simplified):**
```
FCI = Concentration of flow probabilities
    = Measure of how "peaky" vs "flat" the flow distribution is

High FCI: Flows concentrated on few actions (exploitation-heavy)
Low FCI:  Flows spread across many actions (exploration-heavy)
```

**Example:**
```
Action space: [a1, a2, a3, a4, a5]

High FCI (concentrated):
  P(a1)=0.80, P(a2)=0.10, P(a3)=0.05, P(a4)=0.03, P(a5)=0.02
  → FCI ≈ 0.8 (most flow on one action)

Low FCI (dispersed):
  P(a1)=0.21, P(a2)=0.20, P(a3)=0.20, P(a4)=0.20, P(a5)=0.19
  → FCI ≈ 0.2 (flows evenly spread)
```

---

## Pareto Front Smoothness (PFS)

### Definition
**Pareto Front Smoothness (PFS)** measures how continuous and well-behaved the discovered Pareto front is.

### Why It's Important

#### 1. **Loss Ablation Study** ⭐ PRIMARY USE CASE

**File:** `configs/ablations/loss_ablation_final.yaml`

**Why PFS matters here:**
- Different loss functions may produce **jagged** vs. **smooth** Pareto fronts
- Entropy regularization might create gaps in the front
- Credit assignment quality affects front continuity

**Expected insights:**
- **Detailed Balance (DB)**: Very smooth front (local credit assignment converges smoothly)
- **Trajectory Balance (TB)**: Moderately smooth (global credit assignment, some gaps)
- **Flow Matching (FM)**: Potentially jagged (optimization instabilities)
- **SubTB with λ=0.9**: Smooth front (good balance of local/global credit)
- **Too much entropy (β=0.5)**: Jagged front (over-exploration creates gaps)

**Why this is critical:**
A smooth Pareto front indicates:
1. **Convergence quality**: Loss function optimized properly
2. **No sampling gaps**: Algorithm explores uniformly across front
3. **Interpolation confidence**: Can trust interpolation between samples

**Poor PFS (jagged front) indicates:**
- Numerical instabilities in loss computation
- Inadequate exploration of certain preference regions
- Mode collapse or preference collapse

---

#### 2. **Sampling Ablation Study** (Mentioned but not primary)

**File:** `configs/ablations/sampling_ablation.yaml`

**Reference:**
```yaml
# Comment: "Diversity metrics: TDS, MCE, PFS, PAS"
```

**Why PFS matters:**
- **Temperature** affects exploration → impacts front smoothness
- **Greedy sampling**: May create gaps in front (only exploits known regions)
- **Top-k/top-p sampling**: May skip parts of front (sampling artifacts)

**Less critical here** because sampling affects *which* solutions are found, not how they're optimized (which is what PFS measures).

---

### What PFS Measures Technically

**Formula (simplified):**
```
PFS = Σ(deviations from fitted curve)² / (N × Var(y))

Steps:
1. Filter to Pareto-optimal solutions
2. Sort by first objective
3. Fit polynomial curve (degree 2-3)
4. Compute squared deviations from curve
5. Normalize by number of points and variance
```

**Interpretation:**
- **PFS ≈ 0.0**: Perfectly smooth front (matches polynomial)
- **PFS < 0.001**: Excellent smoothness (high-quality front)
- **PFS > 0.01**: Jagged front with gaps or irregularities

**Example:**
```
Smooth front (PFS ≈ 0.0005):
  Points: (0.1,0.9), (0.2,0.85), (0.3,0.78), (0.4,0.69), ...
  → Closely follows y = 1 - 1.1x + 0.1x²

Jagged front (PFS ≈ 0.025):
  Points: (0.1,0.9), (0.2,0.85), [gap], (0.5,0.40), (0.6,0.38), ...
  → Large deviations from fitted curve
```

---

## Summary Table: Where FCI and PFS Matter

| Study | FCI Importance | PFS Importance | Why |
|-------|---------------|---------------|-----|
| **Loss Ablation** | ⭐⭐⭐ PRIMARY | ⭐⭐⭐ PRIMARY | Different losses → different flow distributions and convergence quality |
| **Capacity × Sampling** | ⭐⭐ Secondary | ❌ Not used | Capacity + temperature affect flow concentration |
| **Capacity × Loss** | ⭐⭐ Secondary | ⭐ Tertiary | Interaction: capacity affects which loss spreads flows best |
| **Sampling × Loss** | ⭐⭐ Secondary | ❌ Not used | Sampling strategy + loss function affect exploration |
| **Sampling Ablation** | ❌ Not used | ⭐ Mentioned | Temperature affects front smoothness (but not primary focus) |
| **Capacity Ablation** | ❌ Not used | ❌ Not used | Focused on MCE, PAS, QDS |

---

## Key Insights by Study Type

### Loss Ablation (FCI + PFS both critical)

**Research Questions:**
1. Which loss function produces the most diverse flows? → **FCI**
2. Which loss function produces the smoothest Pareto front? → **PFS**
3. Does entropy regularization spread flows? → **FCI**
4. Does entropy regularization create gaps in the front? → **PFS**

**Expected Findings:**
- **FCI**: Entropy ↓ FCI (spreads flows), SubTB(0.9) moderate FCI (balanced)
- **PFS**: SubTB(0.9) ↓ PFS (smooth), entropy(β>0.1) ↑ PFS (gaps)

---

### Factorial Studies (FCI important, PFS less so)

**Research Questions:**
1. Does high capacity + low temperature cause flow concentration? → **FCI**
2. Do certain loss functions only work with specific capacities? → **FCI**
3. Does sampling strategy mask loss function effects? → **FCI**

**Why PFS is less important:**
Factorials focus on **main effects** and **interactions** on diversity (MCE, PAS), not convergence quality. PFS is more of a diagnostic than a primary outcome.

---

## When to Prioritize FCI

**Use FCI as primary metric when:**
1. ✅ Studying **loss functions** (main driver of flow distribution)
2. ✅ Testing **entropy regularization** (explicitly affects flow concentration)
3. ✅ Analyzing **exploration vs. exploitation** trade-offs
4. ✅ Diagnosing **mode collapse** (high FCI = collapsed flows)

**FCI is less useful when:**
1. ❌ Comparing **non-GFlowNet baselines** (Random, NSGA-II don't have flows)
2. ❌ Studying **preference distributions** (doesn't affect flow mechanics)
3. ❌ Analyzing **final solution quality** (FCI is about process, not outcome)

---

## When to Prioritize PFS

**Use PFS as primary metric when:**
1. ✅ Studying **loss function convergence quality**
2. ✅ Detecting **numerical instabilities** in training
3. ✅ Validating **full Pareto front coverage** (no gaps)
4. ✅ Comparing **credit assignment methods** (TB vs. DB vs. SubTB)

**PFS is less useful when:**
1. ❌ Diversity is already measured by MCE (redundant if MCE is good)
2. ❌ Problem has discrete/non-smooth Pareto front (e.g., some combinatorial problems)
3. ❌ Early in training (front hasn't formed yet)

---

## Recommended Usage

### For Your Keynote Presentation:

**Include FCI when discussing:**
- Loss ablation results
- How different training objectives affect exploration
- Mode collapse analysis

**Include PFS when discussing:**
- Convergence quality of different loss functions
- Why certain configurations produce better Pareto fronts
- Training stability analysis

**Example slide content:**
```
Loss Ablation Key Findings:

FCI (Flow Concentration):
  - SubTB(0.9): Moderate FCI (0.45) → Balanced exploration
  - Entropy(β=0.05): Low FCI (0.32) → Forced exploration ✓
  - DB: High FCI (0.68) → Greedy, mode collapse risk ✗

PFS (Pareto Front Smoothness):
  - SubTB(0.9): Smooth (PFS=0.003) → High-quality front ✓
  - Entropy(β=0.5): Jagged (PFS=0.021) → Over-explored, gaps ✗
  - TB: Moderate (PFS=0.008) → Acceptable quality
```

---

## Bottom Line

**FCI** and **PFS** are **GFlowNet-specific metrics** that are most important in:

1. **Loss Ablation** (PRIMARY) - Different training objectives → different flow distributions and convergence
2. **Factorial Studies** (SECONDARY) - Interactions between factors affect flow concentration

They are **NOT** important in:
- Capacity Ablation (focused on architectural diversity)
- Baseline Comparisons (non-GFlowNet algorithms don't have flows/fronts)

**For your research:**
- Report FCI/PFS prominently in **loss ablation results**
- Mention FCI in **factorial interactions** where relevant
- De-emphasize FCI/PFS in **capacity** and **sampling** ablations (use MCE, PAS, QDS instead)
