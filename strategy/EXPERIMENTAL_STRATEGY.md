# Complete Experimental Strategy

## Overview: Two-Phase Design

### Phase 1: Discovery (HyperGrid) - Weeks 1-8
**Goal**: Systematically identify best practices
**Environment**: HyperGrid (simple, fast, controlled)
**Experiments**: ~250 runs, ~100 hours compute

### Phase 2: Validation (Real Tasks) - Weeks 9-12
**Goal**: Verify findings generalize to practice
**Environments**: 3-grams, Molecules, Sequences
**Experiments**: ~45 runs, ~180 hours compute

**Total**: ~300 runs, ~280 hours compute

---

## Phase 1: Discovery on HyperGrid

### Why HyperGrid First?

✅ **Advantages**:
- **Fast**: 24 min per run vs. 2-6 hours for real tasks (10-20x faster)
- **Simple**: Easy to interpret, clear ground truth
- **Controlled**: Deterministic, reproducible
- **Scalable**: Can run hundreds of experiments
- **Debuggable**: Easy to identify what's working

❌ **Disadvantages**:
- May not capture all complexity of real tasks
- Need validation phase to confirm transfer

💡 **Best Use**: Rapid exploration to narrow down promising approaches

### Experiments in Phase 1

#### Week 1-2: Capacity Ablation
**Question**: How much model capacity do we need?

**Design**: 22 configs × 5 seeds = 110 runs
- Factor: hidden_dim × num_layers × conditioning
- Levels: small, medium, large

**Output**: Winner = Medium (128×4, concat)

**Time**: ~44 hours (110 × 24min)

---

#### Week 3-4: Sampling Ablation
**Question**: How should we explore?

**Design**: 22 configs × 5 seeds = 110 runs (some completed)
- Temperature: 0.5, 1.0, 2.0, 5.0
- Strategy: greedy, categorical, top-k, top-p
- Policy: on-policy, off-policy ratios
- Preferences: uniform, Dirichlet(α)
- Batch size: 32, 64, 128, 256, 512

**Output**: Winner = temp_high (τ=2.0, categorical)

**Time**: ~44 hours

---

#### Week 5-6: Loss Ablation (5 groups)
**Question**: Which loss function works best?

**Design**: 21 configs × 5 seeds = 105 runs
- Group 1: Base losses (TB, DB, SubTB variants, FM) - 30 runs
- Group 2: Entropy regularization (β sweep) - 25 runs
- Group 3: KL regularization - 15 runs
- Group 4: SubTB + Entropy combinations - 20 runs
- Group 5: Loss modifications - 15 runs

**Output**: Expected winner = SubTB(λ=0.9) + Entropy(β=0.05)

**Time**: ~42 hours

**Run sequentially**:
```bash
# Week 5: Groups 1-3
python scripts/run_loss_ablation_group.py --group base_loss_comparison
python scripts/run_loss_ablation_group.py --group entropy_regularization
python scripts/run_loss_ablation_group.py --group kl_regularization

# Week 6: Groups 4-5
python scripts/run_loss_ablation_group.py --group subtb_entropy_sweep
python scripts/run_loss_ablation_group.py --group loss_modifications
```

---

#### Week 7: Capacity × Sampling Factorial
**Question**: Does optimal temperature depend on model size?

**Design**: 9 configs × 5 seeds = 45 runs
- Capacity: small, medium, large
- Temperature: 1.0, 2.0, 5.0

**Output**: Discover if factors interact

**Time**: ~18 hours

```bash
python scripts/run_factorial_experiment.py \
    --config configs/factorials/capacity_sampling_2way.yaml
```

---

#### Week 8: Sampling × Loss Factorial
**Question**: Does optimal loss depend on exploration?

**Design**: 9 configs × 5 seeds = 45 runs
- Temperature: 1.0, 2.0, 5.0
- Loss: TB, SubTB(0.9), SubTB+Entropy

**Output**: Test if better losses need less exploration

**Time**: ~18 hours

```bash
python scripts/run_factorial_experiment.py \
    --config configs/factorials/sampling_loss_2way.yaml
```

---

### Phase 1 Summary

**Total HyperGrid Experiments**: ~250 runs
**Total Time**: ~100 hours sequential, ~10 hours parallel
**Output**: 2-3 promising configurations identified

**Example Winners**:
1. **Overall Winner**: Medium + High temp + SubTB+Entropy
2. **Diversity-Focused**: Large + Very high temp + SubTB+Entropy
3. **Baseline**: Medium + Low temp + TB

---

## Phase 2: Validation on Real Tasks

### Week 9: Analysis & Selection

**Tasks**:
1. Analyze all HyperGrid results
2. Run statistical tests (ANOVA, post-hoc)
3. Create visualizations
4. Select 2-3 configurations to validate
5. Write validation experiment configs

**Deliverables**:
- `analysis/hypergrid_summary.pdf`
- `analysis/statistical_tests.txt`
- `configs/validation/selected_configs.yaml`

---

### Week 10: Validation on 3-Grams

**Task**: Discrete sequence generation
**Why**: Most similar to HyperGrid (discrete, similar length)

**Design**: 3 configs × 5 seeds = 15 runs
**Time**: ~30 hours (2hr per run)

**Configs to test**:
- Winner from HyperGrid
- Diversity-focused (if different)
- Baseline

**Expected outcome**: ✅ Strong transfer (discrete → discrete)

```bash
python scripts/run_validation.py \
    --config configs/validation/validation_3grams.yaml \
    --configs_to_test winner,diversity_focused,baseline
```

---

### Week 11: Validation on Molecules

**Task**: Drug discovery, chemical generation
**Why**: High-impact application, 3 objectives

**Design**: 3 configs × 5 seeds = 15 runs
**Time**: ~60 hours (4hr per run)

**Objectives**:
- QED (drug-likeness)
- SA (synthesizability)
- Diversity (structural)

**Expected outcome**: ⚠️ Partial transfer (scaling to 3 objectives)

```bash
python scripts/run_validation.py \
    --config configs/validation/validation_molecules.yaml \
    --configs_to_test winner,diversity_focused,baseline
```

---

### Week 12: Validation on Sequences (Protein/DNA)

**Task**: Biological sequence design
**Why**: Long sequences, complex constraints

**Design**: 3 configs × 5 seeds = 15 runs
**Time**: ~90 hours (6hr per run)

**Objectives**:
- Stability (folding energy)
- Function (binding affinity)
- Diversity

**Expected outcome**: ⚠️ Moderate transfer (long sequences, harder)

```bash
python scripts/run_validation.py \
    --config configs/validation/validation_sequences.yaml \
    --configs_to_test winner,diversity_focused,baseline
```

---

### Phase 2 Summary

**Total Validation Experiments**: ~45 runs
**Total Time**: ~180 hours sequential, ~18 hours parallel
**Output**: Transfer analysis, generalization claims

---

## Complete Timeline

```
Week 1-2:  Capacity Ablation (HyperGrid)        ████████░░░░
Week 3-4:  Sampling Ablation (HyperGrid)        ░░░░████████
Week 5-6:  Loss Ablation (HyperGrid)            ░░░░░░░░████████
Week 7:    Capacity × Sampling Factorial        ░░░░░░░░░░░░░░████
Week 8:    Sampling × Loss Factorial            ░░░░░░░░░░░░░░░░░░████
Week 9:    Analysis & Config Selection          ░░░░░░░░░░░░░░░░░░░░░░██
Week 10:   Validation: 3-grams                  ░░░░░░░░░░░░░░░░░░░░░░░░████
Week 11:   Validation: Molecules                ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████
Week 12:   Validation: Sequences                ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████
Week 13:   Final Analysis & Paper               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████
```

---

## Resource Summary

### Computational Budget

| Phase | Task | Runs | Time/Run | Total Sequential | Total Parallel (10 jobs) |
|-------|------|------|----------|------------------|-------------------------|
| **Phase 1: Discovery** |
| Week 1-2 | Capacity | 110 | 24m | 44h | 4.4h |
| Week 3-4 | Sampling | 110 | 24m | 44h | 4.4h |
| Week 5-6 | Loss | 105 | 24m | 42h | 4.2h |
| Week 7 | Cap × Samp | 45 | 24m | 18h | 1.8h |
| Week 8 | Samp × Loss | 45 | 24m | 18h | 1.8h |
| **Subtotal** | | **415** | | **166h** | **16.6h** |
| **Phase 2: Validation** |
| Week 10 | 3-grams | 15 | 2h | 30h | 3h |
| Week 11 | Molecules | 15 | 4h | 60h | 6h |
| Week 12 | Sequences | 15 | 6h | 90h | 9h |
| **Subtotal** | | **45** | | **180h** | **18h** |
| **TOTAL** | | **460** | | **346h** | **34.6h** |

### Budget with 10 Parallel Jobs
- **HyperGrid experiments**: ~17 hours wall-clock time
- **Validation experiments**: ~18 hours wall-clock time
- **Total**: ~35 hours wall-clock time (~1.5 days)

### Storage Requirements
- HyperGrid: ~50 MB per run × 415 = ~21 GB
- Validation: ~100 MB per run × 45 = ~4.5 GB
- **Total**: ~26 GB

---

## Expected Scientific Contributions

### 1. Systematic Ablation Study (HyperGrid)
**Contribution**: First comprehensive study of diversity mechanisms in MOGFNs
**Evidence**: 415 experiments testing capacity, sampling, loss
**Claim**: "We systematically evaluate..."

### 2. Factor Interactions (Factorial Experiments)
**Contribution**: First demonstration that factors interact in MOGFNs
**Evidence**: 90 experiments revealing when optimal settings depend on context
**Claim**: "We discover that optimal sampling temperature depends on model capacity..."

### 3. Generalization to Real Tasks (Validation)
**Contribution**: Show findings transfer from toy problem to practice
**Evidence**: 45 validation experiments on 3 domains
**Claim**: "Our approach generalizes to real-world applications including drug discovery..."

### 4. Practical Guidelines
**Contribution**: Actionable recommendations for practitioners
**Evidence**: Decision tree based on task properties
**Claim**: "We provide guidelines for selecting hyperparameters based on task characteristics..."

---

## Risk Mitigation

### Risk 1: HyperGrid findings don't transfer
**Mitigation**:
- Include 3-grams as intermediate complexity
- Analyze what transfers vs. task-specific
- Still valuable: guidelines for when to use which approach

### Risk 2: Time overruns
**Mitigation**:
- HyperGrid phase is fast and predictable
- Can skip Week 12 (sequences) if needed
- Fractional validation (2 configs instead of 3)

### Risk 3: No significant differences found
**Mitigation**:
- Extensive pilot studies suggest differences exist
- Large sample sizes (5 seeds) for statistical power
- Multiple metrics increase chance of detecting effects

### Risk 4: Computational budget exceeded
**Mitigation**:
- Conservative time estimates (actual may be faster)
- Can reduce seeds from 5 to 3 if needed (still publishable)
- Phase 1 results valuable even without Phase 2

---

## Deliverables

### Week 8 Milestone (After HyperGrid)
- [ ] `results/ablations/capacity/all_results.csv`
- [ ] `results/ablations/sampling/all_results.csv`
- [ ] `results/ablations/loss/all_results.csv`
- [ ] `results/factorials/capacity_sampling/results.csv`
- [ ] `results/factorials/sampling_loss/results.csv`
- [ ] `analysis/hypergrid_summary.pdf`
- [ ] Selected 2-3 configurations for validation

### Week 12 Milestone (After Validation)
- [ ] `results/validation/3grams/results.csv`
- [ ] `results/validation/molecules/results.csv`
- [ ] `results/validation/sequences/results.csv`
- [ ] `analysis/transfer_analysis.pdf`
- [ ] Paper draft with all figures

---

## Conclusion

This two-phase strategy combines:
1. **Breadth** (systematic ablations on HyperGrid)
2. **Depth** (factor interactions via factorials)
3. **Validation** (transfer to real tasks)

**Result**: Comprehensive, rigorous, and practical study of diversity in MOGFNs.

**Timeline**: 13 weeks
**Budget**: ~35 hours wall-clock time (with parallelization)
**Output**: High-impact paper with strong experimental evidence
