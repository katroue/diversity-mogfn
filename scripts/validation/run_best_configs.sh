#!/bin/bash
# ============================================================================
# RERUN BEST CONFIGURATIONS WITH CORRECTED PARAMETERS
# ============================================================================
# Purpose: Validate best configurations from factorial studies with
#          off_policy_ratio = 0.0 (corrected from 0.1)
#
# This validates that:
#   1. Factorial-derived configurations work when parameters align
#   2. Predicted QDS scores are achievable
#   3. Mode collapse was due to off-policy interaction, not bad configs
#
# Design: 4 tasks × 5 seeds = 20 experiments
# Time: ~8-12 hours total (varies by task)
#
# Usage:
#   bash scripts/validation/run_best_configs.sh
#
# Or with high priority:
#   sudo nice -n -10 bash scripts/validation/run_best_configs.sh
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Rerun Best Configurations (Corrected Parameters)${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}Experiment Design:${NC}"
echo "  - Tasks: HyperGrid, N-grams, Molecules, Sequences"
echo "  - Seeds: 5 per task"
echo "  - Total: 20 experiments"
echo "  - Change: off_policy_ratio: 0.1 → 0.0"
echo ""
echo -e "${YELLOW}Expected Results:${NC}"
echo "  - HyperGrid:  QDS ~0.60 (was 0.19 with off=0.1)"
echo "  - N-grams:    QDS ~0.58"
echo "  - Molecules:  QDS ~0.66"
echo "  - Sequences:  QDS ~0.60"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo ""

# Record start time
OVERALL_START=$(date +%s)
echo -e "${GREEN}Started at: $(date)${NC}"
echo ""

# Task configurations
declare -A TASKS
TASKS[hypergrid]="configs/factorials/hypergrid_best_config.yaml"
TASKS[ngrams]="configs/factorials/ngrams_best_config.yaml"
TASKS[molecules]="configs/factorials/molecules_best_config.yaml"
TASKS[sequences]="configs/factorials/sequences_best_config.yaml"

declare -A SCRIPTS
SCRIPTS[hypergrid]="scripts/factorials/hypergrid/run_factorial_hypergrid.py"
SCRIPTS[ngrams]="scripts/factorials/ngrams/run_factorial_experiment_ngrams.py"
SCRIPTS[molecules]="scripts/factorials/molecules/run_factorial_molecules.py"
SCRIPTS[sequences]="scripts/factorials/sequences/run_factorial_sequences.py"

# Run each task
SUCCESS_COUNT=0
FAIL_COUNT=0

for TASK in hypergrid ngrams molecules sequences; do
    CONFIG="${TASKS[$TASK]}"
    SCRIPT="${SCRIPTS[$TASK]}"
    OUTPUT_DIR="results/factorials/best_configs/${TASK}_best"

    echo -e "${BLUE}───────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Running: ${TASK}${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────────────────────${NC}"
    echo ""
    echo "  Config: $CONFIG"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    TASK_START=$(date +%s)

    # Run the experiment
    python "$SCRIPT" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"

    EXIT_CODE=$?
    TASK_END=$(date +%s)
    TASK_DURATION=$((TASK_END - TASK_START))
    TASK_MINS=$((TASK_DURATION / 60))

    echo ""

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ $TASK completed successfully in ${TASK_MINS} minutes${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ $TASK failed with exit code $EXIT_CODE${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo ""
done

# Summary
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_HOURS=$((OVERALL_DURATION / 3600))
OVERALL_MINS=$(( (OVERALL_DURATION % 3600) / 60 ))

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}SUMMARY${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}Total Duration: ${OVERALL_HOURS}h ${OVERALL_MINS}m${NC}"
echo -e "${GREEN}Successful tasks: $SUCCESS_COUNT / 4${NC}"

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed tasks: $FAIL_COUNT / 4${NC}"
fi

echo ""
echo -e "${BLUE}Results locations:${NC}"
for TASK in hypergrid ngrams molecules sequences; do
    OUTPUT_DIR="results/factorials/best_configs/${TASK}_best"
    echo "  - $TASK: $OUTPUT_DIR"
done

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Check results: cat results/factorials/best_configs/*/results.csv"
echo "  2. Compare to predictions:"
echo "     - Check if QDS values match factorial predictions"
echo "     - Verify MCE > 0 (no mode collapse)"
echo "  3. Generate report: python scripts/analysis/compare_best_configs.py"
echo ""
echo -e "${BLUE}===============================================================================${NC}"

# Exit with failure if any task failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
