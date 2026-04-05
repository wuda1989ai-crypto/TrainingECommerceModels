#!/usr/bin/env zsh
# =============================================================================
# daily_workflow.sh
# 每日自動化訓練流程
# 流程:
#   1. 呼叫 Gemini 生成新對話 (append 到 master_conversations.jsonl)
#   2. 重新產生 train.jsonl / valid.jsonl
#   3. 備份當前 adapters.safetensors (rollback 後備方案)
#   4. 執行 mlx_lm.lora 訓練 (--steps-per-eval 與 --save-every 對齊為 200)
#   5. 檢查過擬合 (Overfitting Check) —— 若 Val loss 末段明顯高於最低點,
#      自動將 adapters.safetensors 回捲 (rollback) 到最佳 checkpoint
# =============================================================================

set -euo pipefail

# ---- 基本設定 ----------------------------------------------------------------
PROJECT_DIR="/Users/wuda/Python/TrainingECommerceModels"
ADAPTER_DIR="${PROJECT_DIR}/adapters_output"
LOG_FILE="${PROJECT_DIR}/training_log.txt"
ITERS=1000
OVERFIT_THRESHOLD=0.15   # 末段 Val loss 比最低點高出多少視為過擬合
CONDA_ENV="mlx_env"

cd "${PROJECT_DIR}"

# ---- 啟用 Conda 環境 ---------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') 開始每日訓練 =========="

# ---- Step 1. 生成新資料 ------------------------------------------------------
echo "[1/5] 呼叫 Gemini 生成新對話..."
python3 generate_gemini_data.py

echo "[2/5] 產生 train.jsonl / valid.jsonl..."
python3 generate_ecommerce_data.py

# ---- Step 3. 備份目前 adapter (rollback 用後備方案) --------------------------
echo "[3/5] 備份當前 adapters.safetensors..."
BACKUP_FILE="${ADAPTER_DIR}/adapters.safetensors.bak"
if [[ -f "${ADAPTER_DIR}/adapters.safetensors" ]]; then
  cp "${ADAPTER_DIR}/adapters.safetensors" "${BACKUP_FILE}"
fi

# ---- Step 4. 執行 LoRA 訓練 --------------------------------------------------
# 注意: --steps-per-eval 與 --save-every 對齊為 200,確保 Val 最佳點必然有對應 checkpoint,
#       否則過擬合發生時會找不到 {iter:07d}_adapters.safetensors 而退回 fallback 備份。
echo "[4/5] 開始 LoRA 訓練 (${ITERS} iters)..."
python -m mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train \
  --data ./data \
  --iters "${ITERS}" \
  --batch-size 2 \
  --steps-per-report 10 \
  --steps-per-eval 200 \
  --save-every 200 \
  --learning-rate 1e-5 \
  --adapter-path "${ADAPTER_DIR}" >> "${LOG_FILE}" 2>&1

# ---- Step 5. 過擬合檢查 ------------------------------------------------------
echo "[5/5] 檢查過擬合..."

# 擷取「本次訓練」的 Val loss 記錄 (只看 log 尾段最後一次訓練)
# mlx_lm.lora 訓練起始會輸出 "Starting training" 之類的訊息,這裡用
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
  elif [[ -f "${BACKUP_FILE}" ]]; then
    echo "  → 找不到對應 checkpoint,還原訓練前的備份"
    cp "${BACKUP_FILE}" "${ADAPTER_DIR}/adapters.safetensors"
  else
    echo "  → 無可回捲的檔案,保持現狀"
  fi
else
  echo "✅ 未偵測到明顯過擬合,保留本次訓練結果。"
fi

# 清除備份
rm -f "${BACKUP_FILE}"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') 每日訓練結束 =========="
