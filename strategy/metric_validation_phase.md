# Phase 4: Metric Validation (Week 7)

**Goal:** Validate that proposed metrics are useful

**Status:** Planning

---

## Overview

This phase validates the 7 categories of diversity metrics implemented in the project by evaluating their:
- **Distinctiveness:** Do they measure different aspects of diversity?
- **Predictive Power:** Can early metrics predict final outcomes?
- **Consistency:** Do rankings generalize across tasks?
- **Human Alignment:** Do metrics align with human judgment?

---

## Validation 1: Correlation Analysis

**Objective:** Determine if metrics measure different aspects of diversity or are redundant.

### Tasks

#### 1.1 Compute All Metrics on All Experiments
- [x] Metrics already computed for ablation studies (capacity, sampling, loss)
- [x] Metrics already computed for factorial experiments (hypergrid, ngrams, molecules, sequences)
- [ ] Verify all 7 metric categories are present in all results.csv files:
  - Traditional: `hypervolume`, `spacing`, `spread`
  - Trajectory: `tds`, `mpd`
  - Spatial: `mce`, `num_unique_solutions`, `pmd`
  - Objective: `pfs`, `pas`
  - Dynamics: `rbd`
  - Flow: `fci`
  - Composite: `qds`, `der`

#### 1.2 Correlation Matrix Analysis
Create correlation matrices to identify redundant metrics.

**Script to create:**
```bash
scripts/validation/compute_metric_correlations.py
```

**Inputs:**
- `results/ablations/capacity/all_results.csv`
- `results/ablations/sampling/all_results.csv`
- `results/ablations/loss/base_loss_comparison/results.csv`
- `results/factorials/*/results.csv` (all 4 tasks × 3 experiment types)

**Outputs:**
- `results/validation/correlation_matrices/`
  - `ablations_correlation_matrix.pdf` (heatmap)
  - `factorials_correlation_matrix.pdf` (heatmap)
  - `correlation_summary.csv` (highly correlated pairs)

**Analysis questions:**
- Which metrics have correlation |r| > 0.9? (redundant)
- Which metrics have correlation |r| < 0.3? (measuring different aspects)
- Are traditional metrics (HV, spacing) redundant with composite metrics (QDS)?

#### 1.3 Factor Analysis
Perform dimensionality reduction to identify key metric dimensions.

**Script to create:**
```bash
scripts/validation/metric_factor_analysis.py
```

**Methods:**
- Principal Component Analysis (PCA)
- Factor Analysis with Varimax rotation

**Outputs:**
- `results/validation/factor_analysis/`
  - `pca_explained_variance.pdf` (scree plot)
  - `factor_loadings.csv` (which metrics load on which factors)
  - `factor_interpretation.txt` (qualitative interpretation)

**Analysis questions:**
- How many dimensions capture 90% of variance?
- Can we identify interpretable factors (e.g., "coverage" vs "spread" vs "convergence")?
- Which metrics are most representative of each factor?

---

## Validation 2: Predictive Power

**Objective:** Determine if early/intermediate metrics predict final diversity outcomes.

### Tasks

#### 2.1 Define Early vs. Final Metrics
**Early metrics** (computed during training, iteration 5000-10000):
- Dynamics: `rbd` (replay buffer diversity)
- Flow: `fci` (flow concentration index)

**Final metrics** (computed at end, iteration 20000):
- Spatial: `mce`, `num_unique_solutions`
- Composite: `qds`

**Challenge:** Current implementation only saves final metrics. Need to modify training to save intermediate metrics.

**Action items:**
- [ ] Modify `src/models/mogfn_pc.py` to save metrics at intermediate checkpoints (every 5000 iterations)
- [ ] Add `intermediate_metrics.json` output to experiment directories
- [ ] Run subset of experiments (e.g., capacity ablation, 1 seed per config) with intermediate tracking

#### 2.2 Train Predictive Models
Build simple regression models to predict final diversity from early signals + configuration.

**Script to create:**
```bash
scripts/validation/train_predictive_models.py
```

**Model specification:**
```
Final_Diversity ~ Early_RBD + Early_FCI + Capacity + Temperature + Loss_Type
```

**Models to try:**
- Linear Regression (baseline)
- Ridge Regression (regularized)
- Random Forest (non-linear)

**Outputs:**
- `results/validation/predictive_models/`
  - `model_performance.csv` (R², MAE, RMSE for each target metric)
  - `feature_importance.pdf` (which early metrics/configs are most predictive)
  - `prediction_vs_actual.pdf` (scatter plots)

**Analysis questions:**
- Can we predict final QDS from early RBD + configuration? (R² > 0.7?)
- Which configuration factors (capacity, temperature, loss) matter most?
- Do early metrics add value beyond just configuration?

#### 2.3 Evaluate Prediction Accuracy
Use cross-validation to assess generalization.

**Cross-validation strategy:**
- Leave-one-seed-out: Train on 4 seeds, test on 1 (repeat 5 times)
- Leave-one-config-out: Train on N-1 configurations, test on held-out config

**Success criteria:**
- Early metrics should improve prediction vs. config-only baseline
- R² > 0.6 for at least one final diversity metric

---

## Validation 3: Task Consistency

**Objective:** Identify which metrics generalize across tasks vs. are task-specific.

### Tasks

#### 3.1 Cross-Task Ranking Agreement
Compare metric rankings across the 4 tasks (hypergrid, ngrams, molecules, sequences).

**Script to create:**
```bash
scripts/validation/cross_task_consistency.py
```

**Inputs:**
- `results/factorials/sampling_loss/results.csv`
- `results/factorials/ngrams_sampling_loss/results.csv`
- `results/factorials/molecules_sampling_loss/results.csv`
- `results/factorials/sequences_sampling_loss/results.csv`

**Method:**
For each metric (e.g., MCE):
1. Rank configurations by metric value within each task
2. Compute Spearman rank correlation between task pairs
3. Average correlation across all task pairs

**Outputs:**
- `results/validation/task_consistency/`
  - `rank_correlation_matrix.csv` (9×9 configs, correlation across tasks)
  - `metric_generalization_scores.csv` (which metrics have highest cross-task correlation)
  - `task_specific_metrics.txt` (metrics with low cross-task agreement)

**Analysis questions:**
- Which metrics have high rank correlation across tasks? (ρ > 0.7)
- Which metrics are task-specific? (ρ < 0.3)
- Do spatial metrics (MCE, num_unique_solutions) generalize better than flow metrics (FCI)?

#### 3.2 Best Configuration by Task
Identify if "best" configuration varies by task or is consistent.

**Analysis:**
- For each task, identify top-3 configurations by QDS
- Check overlap: Do same configurations win across tasks?

**Outputs:**
- `results/validation/task_consistency/best_configs_by_task.csv`

**Interpretation:**
- If best configs are consistent → configuration insights generalize
- If best configs vary by task → need task-specific tuning

---

## Validation 4: Human Evaluation (if time permits)

**Objective:** Validate that high-scoring solution sets are perceived as more diverse by humans.

### Tasks

#### 4.1 Select Solution Sets for Evaluation
Choose 3-5 contrasting configurations per task to show humans.

**Selection criteria:**
- High QDS (top 10%)
- Low QDS (bottom 10%)
- High MCE but low QDS (diverse modes but poor quality)
- High HV but low MCE (high quality but few modes)

**Total:** 4 tasks × 4 configurations = 16 solution sets

#### 4.2 Design Human Evaluation Interface
Create visualization showing solution sets in objective space.

**Script to create:**
```bash
scripts/validation/visualize_for_humans.py
```

**Visualization:**
- 2D scatter plot of objectives (obj1 vs obj2)
- Color by Pareto membership
- Size by uniqueness (if applicable)

**Outputs:**
- `results/validation/human_eval/solution_sets/`
  - `hypergrid_high_qds.png`
  - `hypergrid_low_qds.png`
  - etc.

#### 4.3 Conduct Human Survey
Present pairs of solution sets and ask: "Which is more diverse?"

**Survey platform:** Google Forms / Qualtrics

**Questions per participant:**
- 8 pairwise comparisons (High QDS vs Low QDS)
- 4 tasks × 2 comparisons each

**Participants:** 10-20 (colleagues, grad students, MTurk if budget allows)

**Outputs:**
- `results/validation/human_eval/survey_responses.csv`

#### 4.4 Analyze Agreement with Metrics
Compare human preferences with metric predictions.

**Script to create:**
```bash
scripts/validation/analyze_human_agreement.py
```

**Analysis:**
- Inter-rater agreement (Fleiss' kappa)
- Correlation: Human preference % vs. QDS difference
- Confusion matrix: Human choice vs. Metric-predicted choice

**Outputs:**
- `results/validation/human_eval/agreement_analysis.pdf`
- `results/validation/human_eval/agreement_summary.txt`

**Success criteria:**
- Human preferences align with QDS predictions in >75% of cases
- Metrics that disagree with humans should be flagged for revision

---

## Implementation Timeline

### Week 7, Day 1-2: Correlation & Factor Analysis
- [x] Verify all metrics are computed
- [ ] Implement `compute_metric_correlations.py`
- [ ] Implement `metric_factor_analysis.py`
- [ ] Run analysis on all datasets
- [ ] Document findings in `results/validation/correlation_analysis_summary.md`

### Week 7, Day 3-4: Predictive Power
- [ ] Modify training to save intermediate metrics (if needed)
- [ ] Run experiments with intermediate tracking (subset)
- [ ] Implement `train_predictive_models.py`
- [ ] Evaluate cross-validated performance
- [ ] Document findings in `results/validation/predictive_analysis_summary.md`

### Week 7, Day 5: Task Consistency
- [ ] Implement `cross_task_consistency.py`
- [ ] Analyze rank correlations across tasks
- [ ] Identify generalizable vs. task-specific metrics
- [ ] Document findings in `results/validation/task_consistency_summary.md`

### Week 7, Day 6-7: Human Evaluation (Optional)
- [ ] Select solution sets for evaluation
- [ ] Generate visualizations
- [ ] Design and deploy survey
- [ ] Collect responses
- [ ] Analyze agreement
- [ ] Document findings in `results/validation/human_eval_summary.md`

---

## Key Deliverables

1. **Correlation matrices** showing metric redundancy/distinctiveness
2. **Factor analysis** identifying key metric dimensions
3. **Predictive models** showing early signals of final diversity
4. **Task consistency report** identifying generalizable metrics
5. **(Optional) Human evaluation** validating metric-human alignment

---

## Success Criteria

- ✅ **Validation 1:** Identify 3-5 key metric dimensions that capture 90% of variance
- ✅ **Validation 2:** Achieve R² > 0.6 in predicting final diversity from early metrics
- ✅ **Validation 3:** Identify at least 3 metrics that generalize across tasks (ρ > 0.7)
- ✅ **Validation 4:** (If conducted) Human agreement with QDS predictions in >75% of cases

---

## Next Steps After Phase 4

Based on validation results:
1. **Reduce metric set:** Keep only non-redundant, generalizable metrics
2. **Update recommendations:** Revise ablation/factorial recommendations based on validated metrics
3. **Publish findings:** Draft metric validation section for paper
4. **Archive redundant metrics:** Document but don't emphasize highly correlated metrics

---

## Notes

- All metrics are already implemented in `src/metrics/`
- Most experiments already have metrics computed
- Main work is analysis scripts, not new experiments
- Human evaluation is optional but valuable for publication

---

## References

- **Correlation analysis:** Use `pandas.DataFrame.corr()` and `seaborn.heatmap()`
- **Factor analysis:** Use `sklearn.decomposition.PCA` and `factor_analyzer` library
- **Predictive models:** Use `sklearn.linear_model`, `sklearn.ensemble.RandomForestRegressor`
- **Rank correlation:** Use `scipy.stats.spearmanr`
