#!/bin/bash

# Configuration defaults
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TP_SIZE=2
SP_SIZE=2
OUTPUT_FILE="experiment_summary.txt"

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --tp-size) TP_SIZE="$2"; shift ;;
        --sp-size) SP_SIZE="$2"; shift ;;
        --output) OUTPUT_FILE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Initialize output file
echo "==========================================================================================" > "$OUTPUT_FILE"
echo "                        SHIFT PARALLELISM EXPERIMENT RUN SUMMARY" >> "$OUTPUT_FILE"
echo "==========================================================================================" >> "$OUTPUT_FILE"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "Model: $MODEL" >> "$OUTPUT_FILE"
echo "TP Size: $TP_SIZE | SP Size: $SP_SIZE" >> "$OUTPUT_FILE"
echo "==========================================================================================" >> "$OUTPUT_FILE"

configs=(
    # threshold long_tokens short_tokens short_requests
#    "0 1024 1024 31"
#    "0 4096 928 31"
#    "0 16384 528 31"
#    "0 32768 1 0"
#    "100000 1024 1024 31"
#    "100000 4096 928 31"
#    "100000 16384 528 31"
#    "100000 32768 1 0"

    "1500 1024 1024 31"
    "1500 4096 928 31"
    "1500 16384 528 31"
    "1500 32768 1 0"

#    "0 16384 1 0"
#    "100000 16384 1 0"
)

# Temporary log file
TMP_LOG="tmp_run_output.log"

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    read -r threshold long_tokens short_tokens short_requests <<< "$config"
    
    echo "------------------------------------------------------------"
    echo "Running Config $((i+1))/${#configs[@]}:"
    echo "  Threshold: $threshold, Long: $long_tokens, Short: $short_tokens, Reqs: $short_requests"
    
    # Run the experiment
    VLLM_ATTENTION_BACKEND=TRITON_ATTN ARCTIC_INFERENCE_ENABLED=1 python batch_token_imbalance_expri.py \
        --workload imbalance \
        --model "$MODEL" \
        --tp-size "$TP_SIZE" \
        --sp-size "$SP_SIZE" \
        --threshold "$threshold" \
        --long-tokens "$long_tokens" \
        --short-tokens "$short_tokens" \
        --short-requests "$short_requests" > "$TMP_LOG" 2>&1
        
    STATUS=$?
    
    # Append header for this run to the output summary
    echo "" >> "$OUTPUT_FILE"
    echo "==========================================================================================" >> "$OUTPUT_FILE"
    echo "CONFIG $((i+1)): Threshold=$threshold, Long=$long_tokens, Short=$short_tokens, Reqs=$short_requests" >> "$OUTPUT_FILE"
    echo "==========================================================================================" >> "$OUTPUT_FILE"
    
    if [ $STATUS -eq 0 ]; then
        echo "  Status: SUCCESS"
        
        # Extract Parallelism mode switched line
        sed -n 's/.*\[Shift Parallel\] //p' "$TMP_LOG" >> "$OUTPUT_FILE"
        
        # Extract Running benchmark workload line
        grep "Running benchmark workload" "$TMP_LOG" >> "$OUTPUT_FILE"
        
        # Extract IMBALANCE WORKLOAD ANALYSIS section
        echo "" >> "$OUTPUT_FILE"
        echo "==================================================" >> "$OUTPUT_FILE"
        sed -n '/IMBALANCE WORKLOAD ANALYSIS/,/Imbalance Ratio:/p' "$TMP_LOG" >> "$OUTPUT_FILE"
        echo "==================================================" >> "$OUTPUT_FILE"
        
        # Extract EXPERIMENTAL METRICS & RESULTS section
        sed -n '/EXPERIMENTAL METRICS & RESULTS/,/Throughput:/p' "$TMP_LOG" >> "$OUTPUT_FILE"
        echo "==================================================" >> "$OUTPUT_FILE"
    else
        echo "  Status: FAILED"
        echo "Status: FAILED (Exit Code: $STATUS)" >> "$OUTPUT_FILE"
        echo "Error output snippet:" >> "$OUTPUT_FILE"
        tail -n 20 "$TMP_LOG" >> "$OUTPUT_FILE"
        echo "==================================================" >> "$OUTPUT_FILE"
    fi
done

# Clean up temp log
rm "$TMP_LOG"

echo "------------------------------------------------------------"
echo "All experiments completed. Summary written to $OUTPUT_FILE"
