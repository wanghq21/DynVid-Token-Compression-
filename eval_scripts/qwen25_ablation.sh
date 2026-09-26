#!/bin/bash

# ============================================================================
# qwen25_vl_eval.sh  (DynVid - Qwen2.5-VL)
#
# DynVid Evaluation Script: Dynamism-Aware Adaptive Video Token Compression
#
# Pipeline (post-projector):
#   Vision Encoder (frozen) -> MLP Projector -> 2D Spatial Merge
#   -> DySeg dynamic segmentation (cosine similarity threshold)
#   -> ERA adaptive budget allocation (Participation Ratio)
#   -> DATS anchor selection (Facility Location + dynamism-aware importance)
#   -> Top-K Soft Fusion (anchor-protected mean)
#   -> LLM (Qwen2.5, M-RoPE) -> (optional) attention-based internal pruning
#
# Configurable parameters:
#   1. avg_retention_ratio      - average retention ratio 
#   2. expand_factor            - expansion factor f_e 
#   3. dyseg_threshold          - DySeg segmentation similarity threshold
#   4. min_segment_num          - minimum number of segments
#   5. group_budget_method      - group budget: pr / uniform
#   6. anchor_method            - anchor selection: facility_location / topk
#   7. topk_fusion              - Soft Fusion K value
#   8. anchor_weight            - anchor weight \gamma in fusion
#   9. query_prune_layer        - LLM internal pruning layer index
#  10. llm_prune_ratio          - LLM internal pruning ratio
#
# Based on lmms-eval framework
#
# Usage:
#   bash qwen25_vl_eval.sh
# ============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARNING

# ############################################################################
#  --  Model Configuration (Qwen2.5-VL specific)
# ############################################################################

# Pretrained model path.
PRETRAINED="Qwen/Qwen2.5-VL-7B-Instruct"
ATTN_IMPLEMENTATION="flash_attention_2"

# ############################################################################
#  --  Video Processing
# ############################################################################

MAX_NUM_FRAMES=32

# ############################################################################
#  --  DynVid Core Parameters
# ############################################################################

#  -- Budget  --
BASE_EXPAND_FACTOR=1.25
BASE_AVG_RETENTION_RATIO=0.1

#  -- DySeg Segmentation  --
BASE_DYSEG_THRESHOLD=0.4
BASE_MIN_SEGMENT_NUM=4
BASE_COMPLEMENTARY_SEGMENT=True

#  -- ERA Group Budget Allocation  --
BASE_GROUP_BUDGET_METHOD=pr

#  -- DATS Anchor Selection  --
BASE_ANCHOR_METHOD=facility_location

#  -- Soft Fusion  --
BASE_TOPK_FUSION=10
BASE_ANCHOR_WEIGHT=0.25

#  -- LLM Internal Pruning  --
BASE_QUERY_PRUNE_LAYER=20
BASE_LLM_PRUNE_RATIO=0.3

# ############################################################################
#  --  Evaluation Tasks
# ############################################################################

TASKS=("videomme" "longvideobench_val_v" "egoschema" "mvbench" "mlvu_test" "mlvu_dev" "lvbench" "tempcompass_multi_choice")

# ############################################################################
#  --  Output Directory
# ############################################################################

OUTPUT_DIR="./eval_results/qwen25_vl_dynvid"
LOG_DIR="./eval_logs/qwen25_vl_dynvid"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# General Run Function
# ============================================================================
run_experiment() {
    local EXP_NAME=$1

    # Budget
    local P_EF=${ABL_EXPAND_FACTOR:-$BASE_EXPAND_FACTOR}
    local P_ARR=${ABL_AVG_RETENTION_RATIO:-$BASE_AVG_RETENTION_RATIO}
    local P_BLR=$(awk "BEGIN {printf \"%.4f\", $P_EF * $P_ARR}")

    # DySeg
    local P_DST=${ABL_DYSEG_THRESHOLD:-$BASE_DYSEG_THRESHOLD}
    local P_MSN=${ABL_MIN_SEGMENT_NUM:-$BASE_MIN_SEGMENT_NUM}
    local P_CS=${ABL_COMPLEMENTARY_SEGMENT:-$BASE_COMPLEMENTARY_SEGMENT}

    # ERA
    local P_GBM=${ABL_GROUP_BUDGET_METHOD:-$BASE_GROUP_BUDGET_METHOD}

    # DATS
    local P_AM=${ABL_ANCHOR_METHOD:-$BASE_ANCHOR_METHOD}

    # Soft Fusion
    local P_TK=${ABL_TOPK_FUSION:-$BASE_TOPK_FUSION}
    local P_AWE=${ABL_ANCHOR_WEIGHT:-$BASE_ANCHOR_WEIGHT}

    # LLM pruning
    local P_QPL=${ABL_QUERY_PRUNE_LAYER:-$BASE_QUERY_PRUNE_LAYER}
    local P_LPR=${ABL_LLM_PRUNE_RATIO:-$BASE_LLM_PRUNE_RATIO}

    local RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}_${EXP_NAME}"
    mkdir -p ${RUN_DIR}

    cat > "${RUN_DIR}/params.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "experiment": "${EXP_NAME}",
    "model": "qwen2_5_vl",
    "params": {
        "expand_factor": ${P_EF},
        "avg_retention_ratio": ${P_ARR},
        "before_LLM_retention_ratio": ${P_BLR},
        "dyseg_threshold": ${P_DST},
        "min_segment_num": ${P_MSN},
        "complementary_segment": ${P_CS},
        "group_budget_method": "${P_GBM}",
        "anchor_method": "${P_AM}",
        "topk_fusion": ${P_TK},
        "anchor_weight": ${P_AWE},
        "query_prune_layer": ${P_QPL},
        "llm_prune_ratio": ${P_LPR}
    }
}
EOF

    echo ""
    echo "============================================"
    echo " Experiment: ${EXP_NAME}"
    echo "============================================"
    echo "  expand_factor:           ${P_EF}"
    echo "  avg_retention_ratio:     ${P_ARR}"
    echo "  before_LLM_retention_ratio:  ${P_BLR}"
    echo "  dyseg_threshold:         ${P_DST}"
    echo "  min_segment_num:         ${P_MSN}"
    echo "  complementary_segment:   ${P_CS}"
    echo "  group_budget_method:     ${P_GBM}"
    echo "  anchor_method:           ${P_AM}"
    echo "  topk_fusion:             ${P_TK}"
    echo "  anchor_weight:           ${P_AWE}"
    echo "  query_prune_layer:       ${P_QPL}"
    echo "  llm_prune_ratio:         ${P_LPR}"
    echo "  Output: ${RUN_DIR}"
    echo "============================================"

    for task in "${TASKS[@]}"; do
        echo " Evaluating: $task"

        TASK_RUN_DIR="${RUN_DIR}/${task}"
        mkdir -p "${TASK_RUN_DIR}"

        accelerate launch --main_process_port 18888 \
            --num_processes 4 \
            -m lmms_eval \
            --model qwen2_5_vl \
            --model_args pretrained=${PRETRAINED},enable_dynvid=True,before_LLM_retention_ratio=${P_BLR},attn_implementation=${ATTN_IMPLEMENTATION},max_num_frames=${MAX_NUM_FRAMES},dyseg_threshold=${P_DST},td_min_segment_num=${P_MSN},td_complementary_segment=${P_CS},group_budget_method=${P_GBM},anchor_method=${P_AM},topk_fusion=${P_TK},anchor_weight=${P_AWE},query_prune_layer=${P_QPL},llm_prune_ratio=${P_LPR} \
            --tasks ${task} \
            --batch_size 1 \
            --output_path "${TASK_RUN_DIR}" \
            --log_samples \
            --log_samples_suffix "${EXP_NAME}" \
            2>&1 | tee "${LOG_DIR}/${EXP_NAME}_${task}_${TIMESTAMP}.log"

        #  -- EgoSchema auto-submission  --
        if [ "$task" = "egoschema" ]; then
            LATEST_SUBMISSION=$(find ${TASK_RUN_DIR} -name "inference_results_egoschema_MC_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
            if [ -n "$LATEST_SUBMISSION" ]; then
                SCORE_FILE="${LATEST_SUBMISSION%.json}_score.txt"
                echo "[EgoSchema] Auto-submitting: ${LATEST_SUBMISSION}"
                python dynvid/validate.py --f "$LATEST_SUBMISSION" 2>&1 | tee "$SCORE_FILE"
            fi
        fi

        echo "   ${EXP_NAME} / ${task} -> ${TASK_RUN_DIR}"
    done
}


# ============================================================================
# Ablation Experiments
# ============================================================================

#  -- Ablation: avg_retention_ratio sweep  --
echo ""
echo "################################################################"
echo "# Ablation: AVG_RETENTION_RATIO"
echo "################################################################"
for ARR in 0.2 0.1 0.075 0.05; do
    ABL_AVG_RETENTION_RATIO=${ARR} run_experiment "R${ARR}"
done

#  -- Ablation: w/o ERA (uniform allocation)  --
# echo ""
# echo "################################################################"
# echo "# Ablation: GROUP_BUDGET_METHOD"
# echo "################################################################"
# ABL_GROUP_BUDGET_METHOD=uniform run_experiment "GBM_uniform"

#  -- Ablation: w/o FL (top-k selection)  --
# echo ""
# echo "################################################################"
# echo "# Ablation: ANCHOR_METHOD"
# echo "################################################################"
# ABL_ANCHOR_METHOD=topk run_experiment "AM_topk"

#  -- Ablation: TOPK_FUSION K value  --
# echo ""
# echo "################################################################"
# echo "# Ablation: TOPK_FUSION"
# echo "################################################################"
# for TK in 1 3 5 10 20; do
#     ABL_TOPK_FUSION=${TK} run_experiment "TK${TK}"
# done

#  -- Ablation: DYSEG_THRESHOLD  --
# echo ""
# echo "################################################################"
# echo "# Ablation: DYSEG_THRESHOLD"
# echo "################################################################"
# for DST in 0.4 0.5 0.55 0.6 0.7; do
#     ABL_DYSEG_THRESHOLD=${DST} run_experiment "DST${DST}"
# done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================"
echo " All experiments complete!"
echo " Results: ${OUTPUT_DIR}"
echo " Logs:    ${LOG_DIR}"
echo "================================================================"