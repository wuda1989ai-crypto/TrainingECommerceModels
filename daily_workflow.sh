#!/usr/bin/env zsh
# =============================================================================
# daily_workflow.sh
# 每日自動化訓練流程
#
# 流程:
#   1. 呼叫 Gemini 生成新對話 (append 到 master_conversations.jsonl)
#   2. 重新產生 train.jsonl / valid.jsonl
#   3. 根據 TRAIN_MODE 準備訓練起點:
#        resume = 延續訓練: 用 --resume-adapter-file 接著既有 adapter 繼續學
#        fuse   = 融合重訓: 先 mlx_lm.fuse 把既有 adapter 烙進 base model,
#                          產生/更新 ./fused_model, 清空 adapter 從零重訓
#   4. 執行 mlx_lm.lora 訓練 (--steps-per-eval 與 --save-every 對齊為 200)
#   5. 過擬合檢查 (Overfitting Check) —— 若 Val loss 末段明顯高於最低點,
#      自動將 adapters.safetensors 回捲 (rollback) 到最佳 checkpoint
#
# 用法:
#   ./daily_workflow.sh                      # 預設 resume
#   TRAIN_MODE=resume ./daily_workflow.sh    # 延續訓練 (日常)
#   TRAIN_MODE=fuse   ./daily_workflow.sh    # 融合 + 重訓 (階段性整理)
# =============================================================================

set -euo pipefail

# ---- 基本設定 ----------------------------------------------------------------
PROJECT_DIR="/Users/wuda/Python/TrainingECommerceModels"
ADAPTER_DIR="${PROJECT_DIR}/adapters_output"
FUSED_MODEL_DIR="${PROJECT_DIR}/fused_model"
LOG_FILE="${PROJECT_DIR}/training_log.txt"
ITERS=600
OVERFIT_THRESHOLD=0.15   # 末段 Val loss 比最低點高出多少視為過擬合
CONDA_ENV="mlx_env"

# 原始 HuggingFace Google 基座模型
BASE_MODEL="mlx-community/gemma-4-e2b-it-4bit"

# ---- 訓練模式 (enum) ---------------------------------------------------------
# zsh 沒有原生 enum, 用 readonly 變數模擬, 比對時都走這幾個常數避免 typo。
readonly MODE_RESUME="resume"   # 延續訓練: --resume-adapter-file 接著既有 adapter 繼續學
readonly MODE_FUSE="fuse"       # 融合重訓: mlx_lm.fuse 烙進 base model 後 random init 重訓
readonly VALID_MODES=("${MODE_RESUME}" "${MODE_FUSE}")

# 可用環境變數覆蓋: TRAIN_MODE=fuse ./daily_workflow.sh
TRAIN_MODE="${TRAIN_MODE:-${MODE_RESUME}}"

if [[ "${TRAIN_MODE}" != "${MODE_RESUME}" && "${TRAIN_MODE}" != "${MODE_FUSE}" ]]; then
  echo "❌ 未知的 TRAIN_MODE='${TRAIN_MODE}' (可用值: ${VALID_MODES[*]})"
  exit 1
fi

cd "${PROJECT_DIR}"

# ---- 啟用 Conda 環境 ---------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') 開始每日訓練 (MODE=${TRAIN_MODE}) =========="

# ---- Step 1. 生成新資料 ------------------------------------------------------
echo "[1/5] 呼叫 Gemini 生成新對話..."
python3 generate_gemini_data.py

echo "[2/5] 產生 train.jsonl / valid.jsonl..."
python3 generate_ecommerce_data.py

# ---- Step 3. 根據 mode 準備訓練起點 ------------------------------------------
#
# 一旦執行過 Mode B, ./fused_model 就成為新的真實基座, 之後所有訓練
# (不論 A/B) 都必須以 fused_model 為 --model, 否則等於把過去學到的知識丟掉。
# 這裡用 fused_model 目錄是否存在作為「當前基座」的判斷依據。
if [[ -d "${FUSED_MODEL_DIR}" ]]; then
  CURRENT_MODEL="${FUSED_MODEL_DIR}"
  echo "[3/5] 偵測到 fused_model, 以其為訓練基座"
else
  CURRENT_MODEL="${BASE_MODEL}"
  echo "[3/5] 使用原始基座模型 ${BASE_MODEL}"
fi

BACKUP_FILE="${ADAPTER_DIR}/adapters.safetensors.bak"
RESUME_ARGS=()

if [[ "${TRAIN_MODE}" == "${MODE_RESUME}" ]]; then
  # -------- resume: 延續訓練 --------
  if [[ -f "${ADAPTER_DIR}/adapters.safetensors" ]]; then
    echo "  → [${MODE_RESUME}] 從現有 adapter 延續訓練 (--resume-adapter-file)"
    RESUME_ARGS=(--resume-adapter-file "${ADAPTER_DIR}/adapters.safetensors")
    # 備份供過擬合 fallback 使用
    cp "${ADAPTER_DIR}/adapters.safetensors" "${BACKUP_FILE}"
  else
    echo "  → [${MODE_RESUME}] 找不到現有 adapter, 視為首次訓練 (random init)"
  fi

else
  # -------- fuse: 融合後重訓 --------
  if [[ -f "${ADAPTER_DIR}/adapters.safetensors" ]]; then
    echo "  → [${MODE_FUSE}] 將現有 adapter 融合進 ${CURRENT_MODEL}"
    # 先寫入 .tmp 再原子替換, 避免 fuse 中途失敗時舊 fused_model 被破壞
    TMP_FUSED="${FUSED_MODEL_DIR}.tmp"
    rm -rf "${TMP_FUSED}"
    python -m mlx_lm fuse \
      --model "${CURRENT_MODEL}" \
      --adapter-path "${ADAPTER_DIR}" \
      --save-path "${TMP_FUSED}"
    rm -rf "${FUSED_MODEL_DIR}"
    mv "${TMP_FUSED}" "${FUSED_MODEL_DIR}"
    CURRENT_MODEL="${FUSED_MODEL_DIR}"

    # 清空舊 adapter: 已烙進 fused_model, 留著只會雙重套用
    rm -f "${ADAPTER_DIR}/adapters.safetensors"
    # fuse 模式刻意不建立 .bak: 任何回捲都不得引用已被融合的舊權重
  else
    echo "  → [${MODE_FUSE}] 沒有現有 adapter 可融合, 直接在 ${CURRENT_MODEL} 上訓新 adapter"
  fi
fi

# ---- Step 4. 執行 LoRA 訓練 --------------------------------------------------
# 注意: --steps-per-eval 與 --save-every 對齊為 200, 確保 Val 最佳點必然有對應 checkpoint,
#       否則過擬合發生時會找不到 {iter:07d}_adapters.safetensors 而退回 fallback 備份。
echo "[4/5] 開始 LoRA 訓練 (${ITERS} iters, model=${CURRENT_MODEL})..."
# 限制 GPU batch size, 防止記憶體不足
export MLX_MAX_BATCH_SIZE=16
# 用 caffeinate 包裹訓練指令, 防止 macOS 睡眠/GPU 降頻導致 Metal command buffer 被殺
caffeinate -dims python -m mlx_lm lora \
  --model "${CURRENT_MODEL}" \
  --train \
  --data ./data \
  --iters "${ITERS}" \
  --batch-size 1 \
  --steps-per-report 10 \
  --steps-per-eval 200 \
  --save-every 200 \
  --learning-rate 1e-5 \
  --max-seq-length 512 \
  --adapter-path "${ADAPTER_DIR}" \
  "${RESUME_ARGS[@]}" >> "${LOG_FILE}" 2>&1

# ---- Step 5. 過擬合檢查 ------------------------------------------------------
echo "[5/5] 檢查過擬合..."

# 擷取「本次訓練」的 Val loss 記錄 (只看 log 尾段最後一次訓練)
# mlx_lm lora 訓練起始會輸出 "Starting training" 之類的訊息,這裡用
# "Trainable parameters" 這行當作本次 run 的起點分隔符,較穩定。
RUN_START_LINE=$(grep -n "Trainable parameters" "${LOG_FILE}" | tail -n 1 | cut -d: -f1)
RUN_START_LINE=${RUN_START_LINE:-1}

VAL_LINES=$(tail -n +"${RUN_START_LINE}" "${LOG_FILE}" | grep -E "Iter [0-9]+: Val loss" || true)

if [[ -z "${VAL_LINES}" ]]; then
  echo "⚠️  找不到本次訓練的 Val loss 記錄,略過過擬合檢查。"
  rm -f "${BACKUP_FILE}"
  exit 0
fi

# 解析成「iter val_loss」兩欄
PARSED=$(echo "${VAL_LINES}" | sed -E 's/.*Iter ([0-9]+): Val loss ([0-9.]+).*/\1 \2/')

BEST_LINE=$(echo "${PARSED}" | sort -k2 -g | head -n 1)
LAST_LINE=$(echo "${PARSED}" | tail -n 1)

BEST_ITER=$(echo "${BEST_LINE}" | awk '{print $1}')
BEST_LOSS=$(echo "${BEST_LINE}" | awk '{print $2}')
LAST_ITER=$(echo "${LAST_LINE}" | awk '{print $1}')
LAST_LOSS=$(echo "${LAST_LINE}" | awk '{print $2}')

echo "  最佳 Val loss: ${BEST_LOSS} @ iter ${BEST_ITER}"
echo "  末次 Val loss: ${LAST_LOSS} @ iter ${LAST_ITER}"

# 比較 (LAST - BEST) 與 OVERFIT_THRESHOLD
DIFF=$(awk -v a="${LAST_LOSS}" -v b="${BEST_LOSS}" 'BEGIN{printf "%.4f", a-b}')
IS_OVERFIT=$(awk -v d="${DIFF}" -v t="${OVERFIT_THRESHOLD}" 'BEGIN{print (d>t)?1:0}')

if [[ "${IS_OVERFIT}" == "1" && "${BEST_ITER}" != "${LAST_ITER}" ]]; then
  echo "⚠️  偵測到過擬合 (ΔVal loss=${DIFF} > ${OVERFIT_THRESHOLD})"
  # 組合最佳 checkpoint 檔名 (7 位數補零)
  BEST_CKPT=$(printf "%s/%07d_adapters.safetensors" "${ADAPTER_DIR}" "${BEST_ITER}")
  if [[ -f "${BEST_CKPT}" ]]; then
    echo "  → 回捲 adapters.safetensors 至 iter ${BEST_ITER}"
    cp "${BEST_CKPT}" "${ADAPTER_DIR}/adapters.safetensors"
  elif [[ "${TRAIN_MODE}" == "${MODE_RESUME}" && -f "${BACKUP_FILE}" ]]; then
    echo "  → 找不到對應 checkpoint, 還原訓練前的備份 [${MODE_RESUME}]"
    cp "${BACKUP_FILE}" "${ADAPTER_DIR}/adapters.safetensors"
  else
    # fuse 模式下禁止回捲到 .bak: 舊 adapter 已烙進 fused_model, 再套回等於雙重加成
    echo "  → 無可回捲 checkpoint, 清除 adapters.safetensors (以 fused_model 作為當前狀態)"
    rm -f "${ADAPTER_DIR}/adapters.safetensors"
  fi
else
  echo "✅ 未偵測到明顯過擬合,保留本次訓練結果。"
fi

# 清除備份
rm -f "${BACKUP_FILE}"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') 每日訓練結束 =========="
