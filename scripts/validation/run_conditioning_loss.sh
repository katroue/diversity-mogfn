#!/bin/bash
# ============================================================================
# RUN CONDITIONING × LOSS INTERACTION VALIDATION
# ============================================================================
# Purpose: Execute the 2-way factorial validation experiment to determine
#          optimal conditioning mechanism (concat vs FiLM) for each loss function
#
# Design: 6 conditions × 3 seeds = 18 experiments
# Time: ~3-4 hours total (12 mins per experiment)
#
# Usage:
#   bash scripts/validation/run_conditioning_loss.sh
#
# Or with high priority:
#   sudo nice -n -10 bash scripts/validation/run_conditioning_loss.sh
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Conditioning × Loss Interaction Validation${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}Experiment Design:${NC}"
echo "  - Task: HyperGrid 32×32"
echo "  - Factors: Conditioning (2) × Loss (3) = 6 conditions"
echo "  - Conditioning: concat, FiLM"
echo "  - Loss: TB, SubTB(λ=0.9), SubTB(λ=0.9)+Entropy"
echo "  - Seeds: 3 per condition"
echo "  - Total: 18 experiments"
echo "  - Estimated time: 3-4 hours"
echo ""
echo -e "${YELLOW}Research Question:${NC}"
echo "  Does the optimal conditioning mechanism depend on the loss function?"
echo ""
echo -e "${YELLOW}Implications:${NC}"
echo "  - If concat is universally better: Keep concat in best_config.yaml files"
echo "  - If FiLM is universally better: Update best_config.yaml to use FiLM"
echo "  - If interaction exists: Use loss-specific conditioning"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo ""

# Configuration
CONFIG="configs/validation/conditioning_loss_interaction.yaml"
OUTPUT_DIR="results/validation/conditioning_loss"
PYTHON_SCRIPT="scripts/factorials/hypergrid/run_factorial_hypergrid.py"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)
echo -e "${GREEN}Started at: $(date)${NC}"
echo ""

# Run the experiment
echo -e "${BLUE}Running validation experiment...${NC}"
python3 "$PYTHON_SCRIPT" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))

echo ""
echo -e "${BLUE}===============================================================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Validation experiment completed successfully!${NC}"
else
    echo -e "${RED}✗ Validation experiment failed with exit code $EXIT_CODE${NC}"
fi

echo ""
echo -e "${GREEN}Duration: ${HOURS}h ${MINUTES}m${NC}"
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Check results: ls -la $OUTPUT_DIR"
echo "  2. View summary: cat $OUTPUT_DIR/results.csv"
echo "  3. Analyze interaction:"
echo "     python3 scripts/validation/analyze_conditioning_loss.py"
echo ""
echo -e "${BLUE}Analysis Steps:${NC}"
echo "  1. Run two-way ANOVA: conditioning × loss on QDS, MCE, hypervolume"
echo "  2. Create interaction plot: Does effect of conditioning depend on loss?"
echo "  3. Post-hoc comparisons:"
echo "     - concat_tb vs film_tb"
echo "     - concat_subtb vs film_subtb"
echo "     - concat_subtb_entropy vs film_subtb_entropy"
echo "  4. Make decision for best_config.yaml files"
echo ""
echo -e "${BLUE}Decision Criteria:${NC}"
echo "  - If concat significantly better (p < 0.05, effect > 0.1):"
echo "    → Keep concat in all best_config.yaml files"
echo "  - If FiLM significantly better (p < 0.05, effect > 0.1):"
echo "    → Update all best_config.yaml files to use FiLM"
echo "  - If significant interaction exists:"
echo "    → Use optimal conditioning for each loss function"
echo "  - Else: Keep concat (simpler implementation)"
echo ""

exit $EXIT_CODE
