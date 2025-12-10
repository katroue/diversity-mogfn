#!/bin/bash
# ============================================================================
# RUN TEMPERATURE × OFF-POLICY INTERACTION VALIDATION
# ============================================================================
# Purpose: Execute the 2-way factorial validation experiment
#
# Design: 6 conditions × 3 seeds = 18 experiments
# Time: ~3-4 hours total (12 mins per experiment)
#
# Usage:
#   bash scripts/validation/run_temp_offpolicy.sh
#
# Or with high priority:
#   sudo nice -n -10 bash scripts/validation/run_temp_offpolicy.sh
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Temperature × Off-Policy Interaction Validation${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}Experiment Design:${NC}"
echo "  - Task: HyperGrid 32×32"
echo "  - Factors: Temperature (3) × Off-Policy (2) = 6 conditions"
echo "  - Seeds: 3 per condition"
echo "  - Total: 18 experiments"
echo "  - Estimated time: 3-4 hours"
echo ""
echo -e "${YELLOW}Key Hypothesis:${NC}"
echo "  Off-policy exploration is beneficial at moderate temperatures"
echo "  but causes MODE COLLAPSE at very high temperatures (τ=5.0)"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo ""

# Configuration
CONFIG="configs/validation/temp_offpolicy_interaction.yaml"
OUTPUT_DIR="results/validation/temp_offpolicy"
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
python "$PYTHON_SCRIPT" \
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
echo "  3. Analyze interaction: python scripts/validation/analyze_temp_offpolicy.py"
echo ""
echo -e "${BLUE}Expected Key Finding:${NC}"
echo "  - temp5_off10 should show MCE ≈ 0.0 (mode collapse)"
echo "  - temp5_off0 should show MCE ≈ 0.37 (healthy diversity)"
echo "  - temp1_off10 should show MCE ≈ 0.45 (improvement from off-policy)"
echo ""

exit $EXIT_CODE
