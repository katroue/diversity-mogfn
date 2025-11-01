#!/bin/bash
# Loss Ablation Study - Quick Execution Script
# Run groups sequentially with analysis pauses in between

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/results/ablations/loss"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "  LOSS ABLATION STUDY - Sequential Group Execution"
echo "================================================================================"
echo ""
echo "This script runs all 5 experiment groups in the recommended order."
echo "You can pause between groups to analyze results before proceeding."
echo ""
echo "Groups:"
echo "  1. base_loss_comparison     (30 runs, ~2h)"
echo "  2. entropy_regularization   (25 runs, ~1.5h)"
echo "  3. kl_regularization        (15 runs, ~45min)"
echo "  4. subtb_entropy_sweep      (20 runs, ~1h)"
echo "  5. loss_modifications       (15 runs, ~45min)"
echo ""
echo "Total: 105 runs across 5 groups"
echo "================================================================================"
echo ""

# Parse command line arguments
DRY_RUN=""
RESUME=""
AUTO_CONTINUE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --auto)
            AUTO_CONTINUE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--resume] [--auto]"
            echo "  --dry-run: Preview without running"
            echo "  --resume: Skip completed experiments"
            echo "  --auto: Don't pause between groups"
            exit 1
            ;;
    esac
done

# Function to run a group
run_group() {
    local group_name=$1
    local group_num=$2
    local total_groups=$3

    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}  GROUP ${group_num}/${total_groups}: ${group_name}${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""

    python "$SCRIPT_DIR/run_loss_ablation_group.py" \
        --group "$group_name" \
        --output_dir "$OUTPUT_DIR" \
        $DRY_RUN \
        $RESUME

    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo -e "${YELLOW}⚠ Group ${group_name} encountered errors (exit code: ${exit_code})${NC}"
        echo "Check logs in $OUTPUT_DIR/$group_name/"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Group ${group_name} completed successfully${NC}"
    fi
}

# Function to analyze results
show_analysis_prompt() {
    local group_name=$1
    local results_file="$OUTPUT_DIR/$group_name/results.csv"

    echo ""
    echo -e "${YELLOW}================================================================================${NC}"
    echo -e "${YELLOW}  ANALYZE RESULTS BEFORE PROCEEDING${NC}"
    echo -e "${YELLOW}================================================================================${NC}"
    echo ""
    echo "Results saved to: $results_file"
    echo ""
    echo "Quick analysis commands:"
    echo ""
    echo "  # View summary statistics"
    echo "  python -c \"import pandas as pd; df=pd.read_csv('$results_file'); print(df.groupby('name')[['hypervolume','tds','mce','pas']].mean())\""
    echo ""
    echo "  # Find best configuration"
    echo "  python -c \"import pandas as pd; df=pd.read_csv('$results_file'); print(df.groupby('name')['qds'].mean().sort_values(ascending=False).head())\""
    echo ""
    echo -e "${YELLOW}================================================================================${NC}"
    echo ""
}

# Main execution
if [ -n "$DRY_RUN" ]; then
    echo "[DRY RUN MODE - No experiments will be executed]"
    echo ""
fi

if [ -n "$RESUME" ]; then
    echo "[RESUME MODE - Will skip completed experiments]"
    echo ""
fi

# Group 1: Base Loss Comparison
run_group "base_loss_comparison" 1 5

if [ -z "$DRY_RUN" ] && [ "$AUTO_CONTINUE" = false ]; then
    show_analysis_prompt "base_loss_comparison"
    echo "📊 Review the results above to determine the best base loss."
    echo "   This will inform Groups 2-5."
    echo ""
    read -p "Press Enter to continue to Group 2..."
fi

# Group 2: Entropy Regularization
run_group "entropy_regularization" 2 5

if [ -z "$DRY_RUN" ] && [ "$AUTO_CONTINUE" = false ]; then
    show_analysis_prompt "entropy_regularization"
    echo "📊 Review entropy regularization effects."
    echo ""
    read -p "Press Enter to continue to Group 3..."
fi

# Group 3: KL Regularization
run_group "kl_regularization" 3 5

if [ -z "$DRY_RUN" ] && [ "$AUTO_CONTINUE" = false ]; then
    show_analysis_prompt "kl_regularization"
    echo "📊 Compare KL vs Entropy regularization."
    echo ""
    read -p "Press Enter to continue to Group 4..."
fi

# Group 4: SubTB + Entropy Sweep
run_group "subtb_entropy_sweep" 4 5

if [ -z "$DRY_RUN" ] && [ "$AUTO_CONTINUE" = false ]; then
    show_analysis_prompt "subtb_entropy_sweep"
    echo "📊 Review SubTB + Entropy combinations."
    echo ""
    read -p "Press Enter to continue to Group 5..."
fi

# Group 5: Loss Modifications
run_group "loss_modifications" 5 5

if [ -z "$DRY_RUN" ]; then
    show_analysis_prompt "loss_modifications"
fi

# Final summary
echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}  ALL GROUPS COMPLETED!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Aggregate all results:"
echo "     python scripts/aggregate_loss_ablation_results.py"
echo ""
echo "  2. Generate visualizations:"
echo "     python scripts/create_loss_ablation_report.py"
echo ""
echo "  3. Run statistical tests:"
echo "     python scripts/analyze_loss_ablation.py"
echo ""
echo -e "${GREEN}================================================================================${NC}"
