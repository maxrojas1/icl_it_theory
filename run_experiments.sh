#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# ==============================================================================
# Script: run_experiments.sh
# Description: Automated execution queue for ICL information-theoretic scaling
#              experiments, including 14B model quantization, DGP figure
#              generation, the four paper evaluation runs, and comparative plots.
# ==============================================================================

# Terminal color formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_stage()   { echo -e "\n${BOLD}${BLUE}=== [STAGE] $1 ===${NC}"; }
log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${BOLD}${GREEN}[SUCCESS]${NC} $1"; }

trap 'log_error "Experiment queue interrupted by user."; exit 130' INT TERM

# Root directory check
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Flags
SKIP_QUANT=false
QUANT_ONLY=false
DRY_RUN=false

show_help() {
    cat << EOF
Usage: ./run_experiments.sh [OPTIONS]

Options:
  --skip-quant    Skip checking or running 14B model quantization.
  --quant-only    Only run the 14B model quantization step and exit.
  --dry-run       Print commands without executing them.
  -h, --help      Display this help message.

Description:
  This script executes the complete experiment queue:
    1. Pre-quantizes Qwen/Qwen2.5-14B to W8A16 if not already cached.
    2. Generates the 1D DGP theoretical sigmoid family figure.
    3. Runs Experiment 1: 7B model entropy scaling (Asymptotic analysis).
    4. Runs Experiment 2: All models scaling at H=0.69 (Asymptotic analysis).
    5. Plots comparative asymptotic figures for (Exp 1 + Exp 2).
    6. Runs Experiment 3: 7B model entropy scaling (Scaling bound analysis).
    7. Runs Experiment 4: All models scaling at H=0.69 (Scaling bound analysis).
    8. Plots comparative scaling bound figures for (Exp 3 + Exp 4).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-quant)
            SKIP_QUANT=true
            shift
            ;;
        --quant-only)
            QUANT_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

execute() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        "$@"
    fi
}

get_latest_run() {
    ls -td results/runs/run_* 2>/dev/null | head -n 1 | sed 's|.*/run_||'
}

START_TIME=$(date +%s)
log_info "Experiment queue started at: $(date)"

# ==============================================================================
# STAGE 1: Model Quantization (Qwen2.5-14B)
# ==============================================================================
if [ "$SKIP_QUANT" = false ]; then
    log_stage "Stage 1: Checking 14B Model Quantization"
    QUANT_PATH="models/Qwen2.5-14B-quantized.w8a16"
    if [ -f "${QUANT_PATH}/config.json" ]; then
        log_info "Cached quantized model found at: ${QUANT_PATH}"
    else
        log_info "Quantized checkpoint not found. Checking llmcompressor..."
        if ! uv run python -c "import llmcompressor" 2>/dev/null; then
            log_warn "Installing llmcompressor into .venv..."
            execute uv pip install llmcompressor
        fi
        log_info "Compressing Qwen/Qwen2.5-14B to W8A16 (~15-20 minutes on RTX A6000)..."
        execute uv run python -m src.utils.quantization --model Qwen/Qwen2.5-14B --quant_bits 8
        log_success "Quantization complete: ${QUANT_PATH}"
    fi

    if [ "$QUANT_ONLY" = true ]; then
        log_success "Quantization-only run completed."
        exit 0
    fi
else
    log_info "Skipping model quantization (--skip-quant specified)."
fi

# ==============================================================================
# STAGE 2: Theoretical DGP Figure
# ==============================================================================
log_stage "Stage 2: Theoretical DGP Figure Generation"
execute uv run python main.py --dgp
log_success "Theoretical DGP figure saved to results/figures/dgp_1d_family.pdf"

# ==============================================================================
# STAGE 3: Experiment 1 - 7B Entropy Scaling (Asymptotic Analysis)
# ==============================================================================
log_stage "Stage 3: Experiment 1 - 7B Model Entropy Scaling"
log_info "Evaluating Qwen-2.5 7B across 10 entropy levels (max_train=300, num_train=64, num_contexts=256)..."
execute uv run python main.py \
    --models 7B \
    --max_train 300 \
    --num_train 64 \
    --num_contexts 256 \
    --num_val 64 \
    --num_entropy_levels 10

RUN_ASYMP_7B=$(get_latest_run)
log_success "Experiment 1 completed. Run ID: ${RUN_ASYMP_7B}"

# ==============================================================================
# STAGE 4: Experiment 2 - Model Ladder Scaling at H=0.69 (Asymptotic Analysis)
# ==============================================================================
log_stage "Stage 4: Experiment 2 - Model Ladder Scaling at H=0.69"
log_info "Evaluating full model ladder (0.5B -> 32B) at H=0.69 (max_train=300, num_train=64, num_contexts=256)..."
execute uv run python main.py \
    --max_train 300 \
    --num_train 64 \
    --num_contexts 256 \
    --num_val 64 \
    --entropies 0.69

RUN_ASYMP_MODELS=$(get_latest_run)
log_success "Experiment 2 completed. Run ID: ${RUN_ASYMP_MODELS}"

# ==============================================================================
# STAGE 5: Comparative Asymptotic Plots (Exp 1 + Exp 2)
# ==============================================================================
log_stage "Stage 5: Comparative Asymptotic Plots"
log_info "Generating comparative plots for runs: ${RUN_ASYMP_7B} and ${RUN_ASYMP_MODELS} at target entropy 0.26..."
execute uv run python main.py \
    --run "$RUN_ASYMP_7B" "$RUN_ASYMP_MODELS" \
    --plot asymptotic \
    --target_entropy 0.26
log_success "Comparative asymptotic figures generated."

# ==============================================================================
# STAGE 6: Experiment 3 - 7B Entropy Scaling for Bound Analysis
# ==============================================================================
log_stage "Stage 6: Experiment 3 - 7B Entropy Scaling (Bound Analysis)"
log_info "Evaluating Qwen-2.5 7B across 10 explicit entropies (max_train=1000, num_train=500, old_geom)..."
execute uv run python main.py \
    --models 7B \
    --max_train 1000 \
    --num_train 500 \
    --num_contexts 100 \
    --num_val 100 \
    --old_geom \
    --entropies 0.36 0.4 0.44 0.47 0.51 0.55 0.58 0.62 0.66 0.69

RUN_SCALING_7B=$(get_latest_run)
log_success "Experiment 3 completed. Run ID: ${RUN_SCALING_7B}"

# ==============================================================================
# STAGE 7: Experiment 4 - Model Ladder Scaling at H=0.69 (Bound Analysis)
# ==============================================================================
log_stage "Stage 7: Experiment 4 - Model Ladder Scaling (Bound Analysis)"
log_info "Evaluating full model ladder (0.5B -> 32B) at H=0.69 (max_train=1000, num_train=500, old_geom)..."
execute uv run python main.py \
    --max_train 1000 \
    --num_train 500 \
    --num_contexts 100 \
    --num_val 100 \
    --old_geom \
    --entropies 0.69

RUN_SCALING_MODELS=$(get_latest_run)
log_success "Experiment 4 completed. Run ID: ${RUN_SCALING_MODELS}"

# ==============================================================================
# STAGE 8: Comparative Scaling Bound Plots (Exp 3 + Exp 4)
# ==============================================================================
log_stage "Stage 8: Comparative Scaling Bound Plots"
log_info "Generating comparative scaling plots for runs: ${RUN_SCALING_7B} and ${RUN_SCALING_MODELS} at target entropy 0.69..."
execute uv run python main.py \
    --run "$RUN_SCALING_7B" "$RUN_SCALING_MODELS" \
    --xlim 1 100 \
    --plot scaling \
    --target_entropy 0.69
log_success "Comparative scaling figures generated."

# ==============================================================================
# SUMMARY
# ==============================================================================
END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))
TOTAL_HOURS=$(awk "BEGIN {printf \"%.2f\", $TOTAL_SECONDS/3600}")

echo -e "\n${BOLD}${GREEN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}               ALL EXPERIMENTS COMPLETED SUCCESSFULLY!                        ${NC}"
echo -e "${BOLD}${GREEN}==============================================================================${NC}"
echo -e "Total Execution Time: ${TOTAL_HOURS} hours (${TOTAL_SECONDS}s)"
echo -e "Executed Runs:"
echo -e "  - Exp 1 (7B Asymptotic):       results/runs/run_${RUN_ASYMP_7B}"
echo -e "  - Exp 2 (Ladder Asymptotic):   results/runs/run_${RUN_ASYMP_MODELS}"
echo -e "  - Exp 3 (7B Bound):            results/runs/run_${RUN_SCALING_7B}"
echo -e "  - Exp 4 (Ladder Bound):        results/runs/run_${RUN_SCALING_MODELS}"
echo -e "\nFigures Location: results/figures/"
echo -e "${BOLD}${GREEN}==============================================================================${NC}\n"
