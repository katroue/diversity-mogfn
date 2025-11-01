# Experimental Flow: From Discovery to Validation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DISCOVERY (HyperGrid)                       │
│                         Weeks 1-8 (~17 hours)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
           ┌────────▼────────┐          ┌────────▼────────┐
           │   ABLATIONS     │          │   FACTORIALS    │
           │  (One at a time)│          │  (Interactions) │
           └────────┬────────┘          └────────┬────────┘
                    │                             │
        ┌───────────┼───────────┐      ┌─────────┴─────────┐
        │           │           │      │                   │
    ┌───▼───┐  ┌───▼───┐  ┌───▼───┐  │   ┌───▼───┐  ┌───▼───┐
    │Capacity│ │Sampling│ │  Loss │  │   │Cap × S│ │ S × L │
    │110 runs│ │110 runs│ │105 runs│ │   │45 runs│ │45 runs│
    └───┬───┘  └───┬───┘  └───┬───┘  │   └───┬───┘  └───┬───┘
        │          │          │       │       │          │
        │  Winner  │  Winner  │Winner │       │  Discover│
        │  Medium  │  High    │SubTB+ │       │Interactions│
        │  128×4   │  Temp    │Entropy│       │          │
        └──────────┴──────────┴───────┘       └──────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   ANALYSIS & SELECTION      │
                    │        Week 9               │
                    │                             │
                    │  Select 2-3 best configs:   │
                    │  • Winner (best QDS)        │
                    │  • Diversity-focused (MCE)  │
                    │  • Baseline (control)       │
                    └──────────────┬──────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────┐
│                  PHASE 2: VALIDATION (Real Tasks)                       │
│                       Weeks 10-12 (~18 hours)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    │  Test 3 configs on 3 tasks  │
                    │  3 × 3 × 5 seeds = 45 runs  │
                    │                             │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
    ┌───▼───┐                  ┌───▼───┐                  ┌───▼───┐
    │3-grams│                  │ Mols  │                  │  Seqs │
    │15 runs│                  │15 runs│                  │15 runs│
    │~30 hrs│                  │~60 hrs│                  │~90 hrs│
    └───┬───┘                  └───┬───┘                  └───┬───┘
        │                          │                          │
        │   Strong transfer?       │   Partial transfer?      │  Transfer?
        │   ✓ Discrete→Discrete    │   ⚠ More complex        │  ⚠ Long
        │                          │     3 objectives         │    sequences
        └──────────────────────────┴──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    TRANSFER ANALYSIS        │
                    │         Week 13             │
                    │                             │
                    │  • Compare rankings         │
                    │  • Measure transfer strength│
                    │  • Identify what transfers  │
                    │  • Create guidelines        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       PAPER SECTIONS        │
                    │                             │
                    │  5.1 Ablations (HyperGrid)  │
                    │  5.2 Factorials (HyperGrid) │
                    │  5.3 Validation (Real)      │
                    │  5.4 Guidelines             │
                    │                             │
                    │  Contributions:             │
                    │  ✓ Systematic study         │
                    │  ✓ Factor interactions      │
                    │  ✓ Generalization proof     │
                    │  ✓ Practical guidelines     │
                    └─────────────────────────────┘
```

## Key Insights

### Why This Flow Works

1. **HyperGrid is a Sandbox**
   - Fast iteration (24 min vs 2-6 hours)
   - Test 415 configurations in ~17 hours
   - Identify promising candidates

2. **Factorials Reveal Interactions**
   - Example: Small models + high temp may fail
   - Can't discover this with ablations alone
   - Prevents recommending bad combinations

3. **Validation Proves Generalization**
   - HyperGrid findings → Real tasks
   - 3 domains of increasing complexity
   - Shows practical applicability

4. **Analysis Provides Value**
   - What transfers: "High temp always helps"
   - What doesn't: "3-obj needs different capacity"
   - Guidelines: "If task X, use config Y"

### Timeline

```
Weeks 1-2:  ████ Capacity Ablation (HyperGrid)
Weeks 3-4:  ████ Sampling Ablation (HyperGrid)
Weeks 5-6:  ████ Loss Ablation (HyperGrid)
Week 7:     ██ Capacity × Sampling Factorial
Week 8:     ██ Sampling × Loss Factorial
            ─────────────────────────────────
Week 9:     ── Analysis & Selection
            ─────────────────────────────────
Week 10:    ████ 3-grams Validation
Week 11:    ████████ Molecules Validation
Week 12:    ████████████ Sequences Validation
Week 13:    ████ Transfer Analysis & Paper
```

### Budget Breakdown

**Phase 1 (Discovery)**:
- 415 runs × 24 min = 166 hours sequential
- With 10 parallel jobs = **17 hours wall-clock**
- Cost: Low (simple task)

**Phase 2 (Validation)**:
- 45 runs × 4 hr avg = 180 hours sequential
- With 10 parallel jobs = **18 hours wall-clock**
- Cost: Higher (complex tasks)

**Total**: ~35 hours wall-clock time (1.5 days)

### Risk Mitigation

If validation shows **weak transfer**:
- ✅ Still valuable: "HyperGrid findings don't fully transfer"
- ✅ Still publishable: Systematic ablations + factorials
- ✅ Still useful: "Task-specific tuning needed"
- ✅ New research: "What makes tasks different?"

If validation shows **strong transfer**:
- 🎉 Strong claim: "Findings generalize across domains"
- 🎉 High impact: Practitioners can use directly
- 🎉 Novel: First to show MOGFN hyperparams transfer

### Expected Outcomes

**Most Likely**: Partial transfer
- Core principles work (high temp, medium capacity)
- Some task-specific tuning needed
- **Paper claim**: "Our approach provides excellent starting point with minor adaptation"

**Best Case**: Strong transfer
- HyperGrid winner is best on all tasks
- **Paper claim**: "Our findings generalize across domains"

**Worst Case**: Task-specific
- Different tasks need different configs
- **Paper claim**: "We characterize when each approach works best"

All outcomes are **publishable and valuable**!

## Files Created

### Core Experiment Configs
```
configs/
├── ablations/
│   ├── capacity_ablation.yaml
│   ├── sampling_ablation.yaml
│   └── loss_ablation.yaml
├── factorials/
│   ├── capacity_sampling_2way.yaml
│   ├── sampling_loss_2way.yaml
│   └── template_factorial.yaml
└── validation/
    ├── validation_template.yaml
    └── README.md
```

### Documentation
```
├── EXPERIMENTAL_STRATEGY.md    # Complete timeline & budget
├── EXPERIMENTAL_FLOW.md        # This file (visual overview)
├── FACTORIAL_EXPERIMENTS_SUMMARY.md
└── configs/
    ├── factorials/
    │   ├── README.md
    │   └── QUICKSTART.md
    └── validation/
        └── README.md
```

### Execution Scripts
```
scripts/
├── run_ablation_study.py           # Main ablation runner
├── run_loss_ablation_group.py      # Loss ablation by group
├── run_loss_ablation.sh            # Automated loss ablation
└── (to create)
    ├── run_factorial_experiment.py # Factorial runner
    └── run_validation.py           # Validation runner
```

## Summary

**You have**: A complete, rigorous experimental design
**Timeline**: 13 weeks
**Budget**: ~35 hours wall-clock (very reasonable!)
**Outcome**: High-impact paper with strong evidence

**Next step**: Complete HyperGrid experiments (Weeks 1-8)!
