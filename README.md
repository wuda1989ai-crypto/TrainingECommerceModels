# 繁體中文電商導購助手 — 訓練筆記

基座模型：`mlx-community/Llama-3.2-3B-Instruct-4bit`（Apple MLX 4bit 量化 Llama-3.2-3B Instruct）
訓練框架：`mlx-lm` LoRA 微調，在 Apple Silicon 上執行。

---

## 1. 環境建立

```zsh
# 安裝 Conda（推薦 Miniforge）
brew install miniforge

# 建立 Python 3.11 環境
conda create -n mlx_env python=3.11 -y
conda init zsh
source ~/.zshrc

# 啟用環境（之後每次開新終端機都要執行）
conda activate mlx_env

# 安裝套件
pip install mlx-lm
pip install coremltools
pip install pandas numpy
pip install -q -U google-genai   # 自動化流程需要
```

---

## 2. 資料架構

```
ecommerce_data.py              ← 手寫種子對話 (raw_conversations) 與同義詞表 (SYNONYM_DICT)
data/master_conversations.jsonl ← Gemini 每日自動生成對話（累加，不會被覆蓋）
generate_ecommerce_data.py     ← 讀取上述兩者，輸出 train.jsonl / valid.jsonl
data/train.jsonl               ← 訓練集（每次重新生成）
data/valid.jsonl               ← 驗證集（每次重新生成，不做增強）
```

**新增手動資料的方式：**

| 資料類型 | 位置 | 格式 |
|---------|------|------|
| 高品質種子對話 | `ecommerce_data.py` → `raw_conversations` | `("user 訊息", "assistant 回覆"),` |
| 批量匯入對話 | `data/master_conversations.jsonl` | `{"user": "...", "assistant": "..."}` 每行一筆 |

加完後執行 `python3 generate_ecommerce_data.py` 重建 train/valid。

---

## 3. 手動訓練流程

### Step 1 — 產生訓練資料

```zsh
conda activate mlx_env
python3 generate_ecommerce_data.py
```

輸出：`data/train.jsonl`、`data/valid.jsonl`

### Step 2 — 執行 LoRA 訓練

```zsh
# 背景執行，log 寫到 training_log.txt
nohup python -m mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train \
  --data ./data \
  --iters 1000 \
  --batch-size 2 \
  --steps-per-report 10 \
  --steps-per-eval 200 \
  --save-every 200 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters_output > training_log.txt 2>&1 &
```

> 注意：`--steps-per-eval` 與 `--save-every` 必須對齊（都設 200），確保最佳 Val loss 對應的 iter 有存 checkpoint，供過擬合回捲使用。

長時間訓練時可開另一個終端機執行 `caffeinate -i` 防止系統睡眠。

### Step 3 — 監看訓練進度

```zsh
tail -f training_log.txt
```

觀察 `Val loss` 變化，若末段 Val loss 明顯高於中段最低點，代表有過擬合。

### Step 4 — 推理測試

```zsh
python -m mlx_lm.generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path ./adapters_output \
  --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一位親切、專業的電商導購助手，會根據用戶的需求給出實用的商品建議，並以問句結尾來引導對話。<|eot_id|><|start_header_id|>user<|end_header_id|>

高CP值的nvidia顯卡<|eot_id|><|start_header_id|>assistant<|end_header_id|>

" \
  --max-tokens 200
```

---

## 4. 自動化每日訓練流程

每天自動執行：Gemini 生成新對話 → 重建資料集 → LoRA 訓練 → 過擬合偵測與回捲。

### 前置設定

**設定 Gemini API Key：**

```zsh
# 加入 ~/.zshrc
export GEMINI_API_KEY='你的_API_KEY'
source ~/.zshrc
```

**確認腳本有執行權限：**

```zsh
chmod +x daily_workflow.sh
```

### 手動執行（測試用）

```zsh
./daily_workflow.sh 2>&1 | tee daily_workflow.log
```

### 設定每日排程（Crontab）

```zsh
crontab -e
```

加入以下內容（每天凌晨 3 點執行）：

```
0 3 * * * cd /Users/wuda/Python/TrainingECommerceModels && ./daily_workflow.sh >> daily_workflow.log 2>&1
```

> ⚠️ cron 的 log 導向 `daily_workflow.log`，不要導向 `training_log.txt`，否則會干擾過擬合偵測的 log 解析。

### 自動化流程說明

| 步驟 | 動作 |
|------|------|
| Step 1 | 呼叫 Gemini API 生成 50 筆新對話，append 到 `master_conversations.jsonl` |
| Step 2 | 重新產生 `train.jsonl` / `valid.jsonl` |
| Step 3 | 備份當前 `adapters.safetensors`（rollback 後備） |
| Step 4 | 執行 LoRA 訓練 1000 iters |
| Step 5 | 解析 `training_log.txt`，比較末次 Val loss 與最低 Val loss，差距 > 0.15 則自動回捲到最佳 checkpoint |

詳細流程說明見 `AUTOMATED_TRAINING.md`。

---

## 5. 重要路徑

| 路徑 | 說明 |
|------|------|
| `adapters_output/adapters.safetensors` | 當前使用的 LoRA 權重 |
| `adapters_output/{iter:07d}_adapters.safetensors` | 各 iter 的 checkpoint |
| `adapters_output/adapter_config.json` | LoRA 設定 |
| `training_log.txt` | 訓練 log（mlx_lm.lora 輸出） |
| `daily_workflow.log` | 每日自動化腳本的執行紀錄 |
