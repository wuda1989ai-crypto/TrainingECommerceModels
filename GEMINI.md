# TrainingECommerceModels 開發指南

本專案旨在為電子商務場景微調大型語言模型(LLM),使其具備專業導購與客戶服務的能力。專案採用 Apple Silicon 優化的 `mlx-lm` 框架,並透過 Gemini API 自動擴充訓練資料。

- 都用繁體中文回答

## 專案概述
- **基礎模型**: `mlx-community/gemma-4-e2b-it-4bit`
- **技術路徑**: 使用 LoRA (Low-Rank Adaptation) 技術進行微調。
- **資料生成**: 結合手寫種子對話 + Gemini API 自動生成 + 純 Python 機械化品質審查 (A/B/C/D 評級)。
- **應用場景**: 涵蓋 12 個主分類(3C/家電/美妝/保健/服飾/精品/母嬰/圖書/家具/運動/居家餐廚/戶外汽機車旅遊),包含購前諮詢、規格比較、售後問題等多種電商導購對話。

## 目錄結構
- `data/master_conversations.jsonl`: Gemini 累積生成的對話池(append-only,長期累加)。
- `data/latest_generated_conversations.jsonl`: 僅本次 Gemini 執行接受的對話(每次覆蓋,方便檢視當次成果)。
- `data/train.jsonl` / `data/valid.jsonl`: 由 `generate_ecommerce_data.py` 從種子 + master 重新產生的訓練/驗證集。
- `adapters_output/`: 訓練過程中生成的 LoRA 適配器權重(`.safetensors`)與配置;包含 `adapters.safetensors`(當前使用)與 `{iter:07d}_adapters.safetensors`(checkpoint)。
- `ecommerce_data.py`: 手寫的種子對話模組(`raw_conversations`、`multi_turn_conversations`、`SYNONYM_DICT`)。
- `generate_gemini_data.py`: 呼叫 Gemini API 生成新對話,內建去重、異常字元過濾與品質評級。
- `generate_ecommerce_data.py`: 將種子 + master 合併並輸出 train/valid JSONL。
- `daily_workflow.sh`: 每日自動化腳本(資料生成 → 重建資料集 → 訓練 → 過擬合回捲)。
- `training_log.txt`: 記錄訓練過程中的損失值(Loss)與效能指標。

## 核心指令

### 1. Gemini 自動生成新對話 (可選,但建議每天執行)
需先設定環境變數 `GEMINI_API_KEY`。產出會 append 到 `data/master_conversations.jsonl`,**只有評級 A 或 B 才會寫入**:
```bash
python3 generate_gemini_data.py
```

### 2. 資料準備
從種子(`ecommerce_data.py`)+ master(Gemini 累積)合併重建 train/valid:
```bash
python3 generate_ecommerce_data.py
```

### 3. 模型訓練 (LoRA)
使用 MLX 進行訓練,參數需與 `daily_workflow.sh` 對齊以利過擬合回捲機制:
```bash
export MLX_MAX_BATCH_SIZE=16 && python -m mlx_lm lora \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --train --data ./data \
  --iters 600 --batch-size 1 --steps-per-report 10 \
  --steps-per-eval 200 --save-every 200 \
  --learning-rate 1e-5 --max-seq-length 512 \
  --adapter-path ./adapters_output
```
*註:`--steps-per-eval` 與 `--save-every` 必須對齊,確保最佳 Val loss 對應的 iter 有 checkpoint,供回捲使用。建議在 Apple Silicon 設備上執行。*

### 4. 模型推理 (測試)
載入微調後的適配器進行對話測試。Gemma 4 使用 `<start_of_turn>` 系列標記:
```bash
python -m mlx_lm.generate \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --adapter-path ./adapters_output \
  --prompt "<start_of_turn>user
我想找適合夏天穿的慢跑鞋<end_of_turn>
<start_of_turn>model
" --max-tokens 200
```

### 5. 每日自動化全流程
```bash
./daily_workflow.sh 2>&1 | tee daily_workflow.log
```
腳本流程:Gemini 生成 → 產生 train/valid → 備份 adapter → `caffeinate -dims` 包裹的 LoRA 訓練 → 過擬合偵測,若 `LAST Val loss - BEST Val loss > 0.15` 自動回捲到最佳 checkpoint。

## 開發規範
- **資料格式**: 訓練資料為 `messages` 樣式 JSONL,例如:
  ```json
  {"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
  ```
  `mlx-lm` 會自動套用基座模型的 chat template,**不要**自己塞入原始 ChatML/特殊 token。
- **Gemini 生成資料的品質門檻**: `generate_gemini_data.py` 的 `evaluate_data_quality` 採純 Python 機械評級,規則包含字數分桶、結尾問句比例、開頭用語集中度、商品類別分布、異常字元偵測等。**只有 A 或 B 級會寫入 master**;C/D 級會直接 `sys.exit(1)`,避免污染資料集。
- **去重策略**: Gemini 產出的 user 訊息會做兩階段去重 — (1) 與 master 既有逐字比對 (2) NFKC + 去標點 + 小寫後 SHA1 hash 比對。多輪對話以第一輪 user 為去重基準。
- **語言純度**: prompt 與後處理白名單嚴格只允許繁中 + 英數 + 常見標點 + emoji,自動攔截簡中、日韓假名、印度語系、程式碼殘渣等。
- **權重管理**: 適配器權重統一存放在 `adapters_output/`,請勿隨意更動目錄結構,以免推理時路徑失效。`daily_workflow.sh` 在訓練前會備份 `adapters.safetensors.bak` 作為回捲 fallback。
- **效能監控**: 訓練時應觀察 `training_log.txt` 中的 `Val loss`,確保模型沒有過擬合(Overfitting)。每日自動化腳本已內建過擬合自動回捲(門檻 `OVERFIT_THRESHOLD=0.15`)。
