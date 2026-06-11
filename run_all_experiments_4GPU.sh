#!/bin/bash
# ============================================================
# PDP Final Project: Complete A/B/C/D Experiment Suite
# ============================================================
#
# Runs FOUR modes:
#   1. Pure TP  (TP=4, SP=1, no shift)
#   2. Pure SP  (TP=1, SP=4, no shift, threshold=0)
#   3. Static Shift Parallel  (original threshold logic)
#   4. Adaptive Shift Parallel (our improvement)
#
# For V100 32GB กั 4 GPUs
# ============================================================

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP_SIZE=2
SP_SIZE=2
THRESHOLD="${THRESHOLD:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  PDP Final Project กX Complete Experiment Suite"
echo "  Model: $MODEL"
echo "  TP=$TP_SIZE, SP=$SP_SIZE, Threshold=$THRESHOLD"
echo "  Max Model Len: $MAX_MODEL_LEN"
echo "  Results: $RESULTS_DIR/"
echo "  Time: $(date)"
echo "============================================================"

# Helper function
run_exp() {
    local label="$1"
    local logfile="$2"
    shift 2
    echo ""
    echo "--- $label ---"
    VLLM_ATTENTION_BACKEND=TRITON_ATTN ARCTIC_INFERENCE_ENABLED=1 python batch_token_imbalance_expri.py "$@" \
        2>&1 | tee "$RESULTS_DIR/$logfile"
    echo ""
}

# ============================================================
# GROUP A: Decode-Heavy Workload
# Many short decode requests ก๗ adaptive threshold should keep TP
# ============================================================
echo ""
echo "========== GROUP A: Decode-Heavy (500 short requests) =========="

run_exp "A1: Pure TP (TP=4, SP=1)" "A1_decode_pure_tp.log" \
    --model "$MODEL" --tp-size 4 --sp-size 1 \
    --threshold 99999 --max-model-len $MAX_MODEL_LEN \
    --workload decode --tokens-count 500 \
    --no-shift 
    #--no-spec

run_exp "A2: Pure SP (TP=1, SP=4, no shift)" "A2_decode_pure_sp.log" \
    --model "$MODEL" --tp-size 1 --sp-size 4 \
    --threshold 0 --max-model-len $MAX_MODEL_LEN \
    --workload decode --tokens-count 500 \
    --no-shift 
    #--no-spec

ARCTIC_ADAPTIVE_THRESHOLD=0 \
run_exp "A3: Static Shift (original)" "A3_decode_static_shift.log" \
    --model "$MODEL" --tp-size $TP_SIZE --sp-size $SP_SIZE \
    --threshold $THRESHOLD --max-model-len $MAX_MODEL_LEN \
    --workload decode --tokens-count 500 \
    #--no-spec

ARCTIC_ADAPTIVE_THRESHOLD=1 \
run_exp "A4: Adaptive Shift (ours)" "A4_decode_adaptive_shift.log" \
    --model "$MODEL" --tp-size $TP_SIZE --sp-size $SP_SIZE \
    --threshold $THRESHOLD --max-model-len $MAX_MODEL_LEN \
    --workload decode --tokens-count 500 \
    #--no-spec

# ============================================================
# GROUP B: Imbalanced Workloads (varying long request length)
# 1 long request + 31 short requests
# ============================================================
echo ""
echo "========== GROUP B: Imbalanced Workloads =========="

for LONG_TOKENS in 1024 4096 8192 16384; do
    SHORT_TOKENS=128
    SHORT_REQS=31

    echo ""
    echo ">>> Imbalance: long=${LONG_TOKENS}, short=${SHORT_TOKENS} กั ${SHORT_REQS} <<<"

    run_exp "B-${LONG_TOKENS}: Pure TP (TP=4, SP=1)" \
        "B_${LONG_TOKENS}_pure_tp.log" \
        --model "$MODEL" --tp-size 4 --sp-size 1 \
        --threshold 99999 --max-model-len $MAX_MODEL_LEN \
        --workload imbalance \
        --long-tokens $LONG_TOKENS --short-tokens $SHORT_TOKENS --short-requests $SHORT_REQS \
        --no-shift 
        #--no-spec

    run_exp "B-${LONG_TOKENS}: Pure SP (TP=1, SP=4)" \
        "B_${LONG_TOKENS}_pure_sp.log" \
        --model "$MODEL" --tp-size 1 --sp-size 4 \
        --threshold 0 --max-model-len $MAX_MODEL_LEN \
        --workload imbalance \
        --long-tokens $LONG_TOKENS --short-tokens $SHORT_TOKENS --short-requests $SHORT_REQS \
        --no-shift 
        #--no-spec

    ARCTIC_ADAPTIVE_THRESHOLD=0 \
    run_exp "B-${LONG_TOKENS}: Static Shift" \
        "B_${LONG_TOKENS}_static_shift.log" \
        --model "$MODEL" --tp-size $TP_SIZE --sp-size $SP_SIZE \
        --threshold $THRESHOLD --max-model-len $MAX_MODEL_LEN \
        --workload imbalance \
        --long-tokens $LONG_TOKENS --short-tokens $SHORT_TOKENS --short-requests $SHORT_REQS \
        #--no-spec

    ARCTIC_ADAPTIVE_THRESHOLD=1 \
    run_exp "B-${LONG_TOKENS}: Adaptive Shift" \
        "B_${LONG_TOKENS}_adaptive_shift.log" \
        --model "$MODEL" --tp-size $TP_SIZE --sp-size $SP_SIZE \
        --threshold $THRESHOLD --max-model-len $MAX_MODEL_LEN \
        --workload imbalance \
        --long-tokens $LONG_TOKENS --short-tokens $SHORT_TOKENS --short-requests $SHORT_REQS \
        #--no-spec
done

# ============================================================
# GROUP C: Balanced Workload (sanity check, should show no difference)
# 32 requests กั 1024 tokens each
# ============================================================
echo ""
echo "========== GROUP C: Balanced Workload (32 กั 1024) =========="

run_exp "C1: Pure TP (TP=4, SP=1)" "C1_balanced_pure_tp.log" \
    --model "$MODEL" --tp-size 4 --sp-size 1 \
    --threshold 99999 --max-model-len $MAX_MODEL_LEN \
    --workload imbalance \
    --long-tokens 1024 --short-tokens 1024 --short-requests 31 \
    --no-shift 
    #--no-spec

run_exp "C2: Pure SP (TP=1, SP=4)" "C2_balanced_pure_sp.log" \
    --model "$MODEL" --tp-size 1 --sp-size 4 \
    --threshold 0 --max-model-len $MAX_MODEL_LEN \
    --workload imbalance \
    --long-tokens 1024 --short-tokens 1024 --short-requests 31 \
    --no-shift 
    #--no-spec

ARCTIC_ADAPTIVE_THRESHOLD=0 \
run_exp "C3: Static Shift" "C3_balanced_static_shift.log" \
    --model "$MODEL" --tp-size $TP_SIZE --sp-size $SP_SIZE \
    --threshold $THRESHOLD --max-model-len $MAX_MODEL_LEN \
    --workload imbalance \
    --long-tokens 1024 --short-tokens 1024 --short-requests 31 \
    #--no-spec

ARCTIC_ADAPTIVE_THRESHOLD=1 \
run_exp "C4: Adaptive Shift" "C4_balanced_adaptive_shift.log" \
    --model "$MODEL" --tp-size $TP_SIZE --sp-size $SP_SIZE \
    --threshold $THRESHOLD --max-model-len $MAX_MODEL_LEN \
    --workload imbalance \
    --long-tokens 1024 --short-tokens 1024 --short-requests 31 \
    #--no-spec

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results: $RESULTS_DIR/"
echo "  Time: $(date)"
echo "============================================================"

SUMMARY_FILE="$RESULTS_DIR/4GPU_experiments_summary.log"

{
    echo ""
    echo "Quick summary:"
    echo "--------------------------------------------------------------"
    printf "%-45s %s\n" "4GPU Experiment" "Throughput"
    echo "--------------------------------------------------------------"

    for f in "$RESULTS_DIR"/*.log; do
        name=$(basename "$f" .log)
        tp=$(grep -oP 'Throughput:\s+\K[\d.]+' "$f" 2>/dev/null || echo "N/A")
        printf "%-45s %s tok/s\n" "$name" "$tp"
    done

    echo "--------------------------------------------------------------"
} | tee "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
