# Validation Experiments - Transfer to Real-World Tasks

## Experimental Design Strategy

### Phase 1: Systematic Testing on HyperGrid (Weeks 1-8)
✅ **Fast, cheap, controlled environment**
- Capacity ablation → identify best model size
- Sampling ablation → identify best exploration strategy
- Loss ablation → identify best training objective
- Factorial experiments → discover factor interactions

**Output**: 2-3 promising configurations identified

### Phase 2: Validation on Real Tasks (Weeks 9-10) ← YOU ARE HERE
🎯 **Test generalization to practical applications**
- 3-grams (discrete sequences, text-like)
- Molecule generation (chemical properties)
- Sequence design (protein/biological sequences)

**Goal**: Verify that findings from HyperGrid transfer to real-world problems

## Why This Strategy Works

### Benefits of HyperGrid for Discovery
1. **Fast iteration**: ~24 min per run vs hours for molecules
2. **Controllable**: Clear ground truth, simple reward structure
3. **Interpretable**: Easy to visualize objective space
4. **Reproducible**: Deterministic environment
5. **Scalable**: Can run hundreds of experiments

### Benefits of Real Tasks for Validation
1. **Practical relevance**: Actual use cases
2. **Robustness testing**: Verify generalization
3. **Paper contribution**: Show real-world impact
4. **Failure modes**: Discover limitations

### Risks of Skipping HyperGrid Phase
❌ Running experiments on complex tasks first:
- Takes 10-20x longer → fewer experiments possible
- Harder to debug → unclear what's causing differences
- More confounds → task-specific quirks mask general patterns
- Expensive → limited budget means fewer seeds, less statistical power

## Validation Tasks

### Task 1: 3-Grams (Discrete Sequences)
**Domain**: Text/sequence generation
**Objectives**:
- Objective 1: Fluency (valid transitions)
- Objective 2: Diversity (vocabulary richness)

**Why this task**:
- Similar to HyperGrid (discrete actions)
- Tests generalization from grid navigation to sequence generation
- Practical application in NLP

**Expected transfer**: ✅ HIGH (both discrete, similar structure)

### Task 2: Molecule Generation
**Domain**: Drug discovery, chemical design
**Objectives**:
- Objective 1: QED (drug-likeness)
- Objective 2: SA (synthesizability)
- Objective 3: Diversity (structural variety)

**Why this task**:
- Real-world application with high impact
- Tests scaling to 3 objectives (HyperGrid was 2)
- Continuous molecular properties from discrete graphs

**Expected transfer**: ⚠️ MEDIUM (more complex, 3 objectives)

### Task 3: Sequence Design (Protein/DNA)
**Domain**: Computational biology
**Objectives**:
- Objective 1: Stability (folding energy)
- Objective 2: Function (binding affinity)
- Objective 3: Diversity (sequence variety)

**Why this task**:
- Real scientific application
- Long sequences (harder than HyperGrid)
- Tests biological constraints

**Expected transfer**: ⚠️ MEDIUM-LOW (long sequences, complex constraints)

## Validation Experiment Design

### Selection Criteria
From factorials, select **2-3 configurations**:

**Configuration A: "Winner"**
- Best overall QDS across HyperGrid experiments
- Expected to work well on all tasks

**Configuration B: "Diversity-Focused"** (if different from A)
- Best MCE/TDS on HyperGrid
- May sacrifice some quality for exploration
- Test if diversity emphasis transfers

**Configuration C: "Baseline"**
- Standard settings (Medium + Low temp + TB)
- Control to measure improvement

### Validation Protocol

**For each configuration × each task**:
1. Run **5 seeds** for statistical robustness
2. Evaluate on **same metrics**: MCE, TDS, Hypervolume, QDS
3. Compare to task-specific baselines
4. Analyze which aspects transfer vs. task-specific tuning needed

**Total validation runs**: 3 configs × 3 tasks × 5 seeds = **45 runs**

**Time estimate**:
- 3-grams: ~2 hours per run → 30 hours
- Molecules: ~4 hours per run → 60 hours
- Sequences: ~6 hours per run → 90 hours
- **Total**: ~180 hours sequential, ~18 hours parallel (10 jobs)

## Expected Outcomes

### Scenario 1: Strong Transfer ✅
**Result**: Winner from HyperGrid is also best on all real tasks
**Implication**: HyperGrid findings are general and robust
**Paper claim**: "Our approach generalizes across domains"

### Scenario 2: Partial Transfer ⚠️
**Result**: Winner from HyperGrid works well on 3-grams, needs tuning for molecules/sequences
**Implication**: Core principles transfer, but scaling/domain adaptation needed
**Paper claim**: "Our approach provides good starting point, with task-specific refinement"

### Scenario 3: Task-Specific Patterns 📊
**Result**: Different configs win on different tasks
**Implication**: Need to characterize when each approach works
**Paper claim**: "We provide guidelines for selecting approach based on task properties"

### Scenario 4: Failure to Transfer ❌
**Result**: HyperGrid winner doesn't work well on real tasks
**Implication**: HyperGrid may not be representative enough
**Action**: Investigate why, add intermediate-complexity tasks

## Success Criteria

### Minimum Success
- At least one HyperGrid config transfers to at least 2/3 tasks
- Improvement over baseline on at least 1 diversity metric
- Statistical significance (p < 0.05) on at least 1 task

### Good Success
- HyperGrid winner is competitive (top 2) on all tasks
- Consistent patterns in what transfers (e.g., "high temp always helps")
- Clear recommendations for practitioners

### Excellent Success
- HyperGrid winner achieves SOTA on real tasks
- Discover new insights (e.g., "molecules need higher capacity than expected")
- Validation leads to new research questions

## Configuration Files

This directory contains:
- `validation_3grams.yaml` - 3-gram sequence generation
- `validation_molecules.yaml` - Molecule generation
- `validation_sequences.yaml` - Protein/DNA sequence design
- `template_validation.yaml` - Template for new validation tasks

## Integration with Main Study

### Timeline
**Weeks 1-6**: Ablation studies on HyperGrid
**Weeks 7-8**: Factorial experiments on HyperGrid
**Week 9**: Analyze all results, select 2-3 best configs
**Week 10**: Validation on 3-grams
**Week 11**: Validation on molecules
**Week 12**: Validation on sequences
**Week 13**: Final analysis, paper writing

### Budget Allocation
- HyperGrid experiments: ~250 runs × 24min = 100 hours (cheap exploration)
- Validation experiments: ~45 runs × 4hr avg = 180 hours (focused validation)
- **Total**: ~280 hours (reasonable for major study)

### Decision Points

**After Week 9 Analysis**:
- Select 2-3 configs to validate
- May discover unexpected patterns requiring follow-up
- Budget allows ~1 week buffer for additional experiments

**After Week 10 (3-grams)**:
- Quick check: Is anything transferring?
- If yes → proceed to molecules/sequences
- If no → investigate why, may need intermediate tasks

**After Week 11 (molecules)**:
- Pattern emerging? Adjust sequence experiments if needed
- May skip sequences if budget tight and patterns clear

## Paper Structure

### Results Section
**5.1 Ablation Studies (HyperGrid)**
- Capacity effects
- Sampling effects
- Loss function effects

**5.2 Factorial Experiments (HyperGrid)**
- Interaction between capacity and sampling
- Interaction between sampling and loss
- Optimal configurations identified

**5.3 Validation on Real-World Tasks** ← This section
- Transfer to 3-grams
- Transfer to molecules
- Transfer to sequences
- Analysis of what transfers vs. task-specific tuning

**5.4 Practical Guidelines**
- When to use which configuration
- Task characteristics that predict optimal settings

### Key Contributions
1. Systematic ablation study on HyperGrid (comprehensive)
2. First factorial study of interactions in MOGFNs (novel)
3. Validation showing generalization to real tasks (impactful)
4. Practical guidelines for practitioners (useful)

## Next Steps

1. **Complete HyperGrid experiments** (Weeks 1-8)
2. **Analyze and select configs** (Week 9)
3. **Create validation experiment configs** using templates here
4. **Run validation experiments** (Weeks 10-12)
5. **Compare and analyze** transfer patterns
6. **Write paper** with complete story from discovery to validation
