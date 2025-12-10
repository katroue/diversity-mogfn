#!/bin/bash
# ============================================================================
# RERUN BEST CONFIGURATIONS - SIMPLIFIED VERSION
# ============================================================================
# Purpose: Validate best configurations with corrected parameters
# Compatible with all shell versions
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Rerun Best Configurations (Corrected Parameters)${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo "  - HyperGrid:  QDS ~0.60 (was 0.19 with off=0.1)"
echo "  - N-grams:    QDS ~0.58"
echo "  - Molecules:  QDS ~0.66"
echo "  - Sequences:  QDS ~0.60"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo ""

START_TIME=$(date +%s)

# Task 1: HyperGrid
echo -e "${BLUE}Running HyperGrid...${NC}"
python3 scripts/factorials/hypergrid/run_factorial_hypergrid.py \
    --config configs/factorials/hypergrid_best_config.yaml \
    --output_dir results/validation/hypergrid_best_corrected

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ HyperGrid completed${NC}"
else
    echo -e "${RED}✗ HyperGrid failed${NC}"
fi
echo ""

# Task 2: N-grams
echo -e "${BLUE}Running N-grams...${NC}"
python3 scripts/factorials/ngrams/run_factorial_experiment_ngrams.py \
    --config configs/factorials/ngrams_best_config.yaml \
    --output_dir results/validation/ngrams_best_corrected

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ N-grams completed${NC}"
else
    echo -e "${RED}✗ N-grams failed${NC}"
fi
echo ""

# Task 3: Molecules
echo -e "${BLUE}Running Molecules...${NC}"
python3 scripts/factorials/molecules/run_factorial_molecules.py \
    --config configs/factorials/molecules_best_config.yaml \
    --output_dir results/validation/molecules_best_corrected

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Molecules completed${NC}"
else
    echo -e "${RED}✗ Molecules failed${NC}"
fi
echo ""

# Task 4: Sequences
echo -e "${BLUE}Running Sequences...${NC}"
python3 scripts/factorials/sequences/run_factorial_sequences.py \
    --config configs/factorials/sequences_best_config.yaml \
    --output_dir results/validation/sequences_best_corrected

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Sequences completed${NC}"
else
    echo -e "${RED}✗ Sequences failed${NC}"
fi
echo ""

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(( (DURATION % 3600) / 60 ))

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${GREEN}Completed in ${HOURS}h ${MINS}m${NC}"
echo ""
echo "Check results:"
echo "  cat results/validation/hypergrid_best_corrected/results.csv"
echo "  cat results/validation/ngrams_best_corrected/results.csv"
echo "  cat results/validation/molecules_best_corrected/results.csv"
echo "  cat results/validation/sequences_best_corrected/results.csv"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
