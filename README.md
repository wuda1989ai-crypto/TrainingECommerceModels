# 繁體中文電商導購助手 — 訓練筆記

基座模型：`mlx-community/gemma-4-e2b-it-4bit`（Apple MLX 4bit 量化 Gemma 4 E2B Instruct）
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

# 安裝核心套件
pip install mlx-lm
pip install coremltools
pip install pandas numpy

# 安裝 Gemini 自動生成資料所需套件 (generate_gemini_data.py / daily_workflow.sh 必備)
pip install -q -U google-genai
```

> 註:`google-genai` 是 Google 新版 Python SDK,對應 `from google import genai` 這支 import。若未安裝,`generate_gemini_data.py` 啟動時會直接 `sys.exit("❌ 請先安裝新版 SDK: pip install -q -U google-genai")`。

---

## 2. 資料架構

```
ecommerce_data.py                          ← 手寫種子對話 (raw_conversations、multi_turn_conversations) 與同義詞表 (SYNONYM_DICT)
generate_gemini_data.py                    ← 呼叫 Gemini API 生成新對話 + 去重 + 異常字元過濾 + Python 機械化品質評級
data/master_conversations.jsonl            ← Gemini 累積生成的對話池(append-only,不會被覆蓋)
data/latest_generated_conversations.jsonl  ← 僅本次 Gemini 執行接受的對話(每次覆蓋,方便檢視當次成果)
generate_ecommerce_data.py                 ← 讀取種子 + master,輸出 train.jsonl / valid.jsonl
data/train.jsonl                           ← 訓練集(每次重新生成,單輪會做同義詞增強)
data/valid.jsonl                           ← 驗證集(每次重新生成,不做增強)
```

**新增手動資料的方式:**

| 資料類型 | 位置 | 格式 |
|---------|------|------|
| 高品質種子單輪對話 | `ecommerce_data.py` → `raw_conversations` | `("user 訊息", "assistant 回覆"),` |
| 高品質種子多輪對話 | `ecommerce_data.py` → `multi_turn_conversations` | `[("user", "..."), ("assistant", "..."), ...]` |
| 批量匯入對話 | `data/master_conversations.jsonl` | 單輪 `{"user": "...", "assistant": "..."}` 或多輪 `{"turns": [{"user": "...", "assistant": "..."}, ...]}`,每行一筆 |

加完後執行 `python3 generate_ecommerce_data.py` 重建 train/valid。

**Gemini 自動生成的品質門檻:** `generate_gemini_data.py` 內建 Python 機械化評級(`evaluate_data_quality`),統計字數分桶、結尾問句比例、開頭用語集中度、商品類別分布、異常字元等指標,評為 A/B/C/D 四級。**只有 A 或 B 級會寫入 `master_conversations.jsonl`**,C/D 級會直接 `sys.exit(1)` 放棄,避免污染資料集。產出前還會做兩階段去重(精確字串 + NFKC 正規化 SHA1 hash)以及異常字元白名單過濾(只允許繁中 + 英數 + 常見標點 + emoji)。

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
  export MLX_MAX_BATCH_SIZE=16 && nohup python -m mlx_lm lora \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --train \
  --data ./data \
  --iters 600 \
  --batch-size 1 \
  --steps-per-report 10 \
  --steps-per-eval 200 \
  --save-every 200 \
  --learning-rate 1e-5 \
  --max-seq-length 512 \
  --adapter-path ./adapters_output > training_log.txt 2>&1 &
```

**指令參數說明：**
- `export MLX_MAX_BATCH_SIZE=16`：限制 GPU batch size,防止記憶體不足。
- `--model`：指定基座模型名稱或路徑。
- `--train`：啟用訓練模式。
- `--data`：訓練資料集所在目錄（內需包含 `train.jsonl` 與 `valid.jsonl`）。
- `--iters`：總訓練疊代次數（Iterations）設定為 600 次。
- `--max-seq-length`：截斷序列上限為 512 token,搭配 `batch-size 1` 控制記憶體使用。
- `--batch-size`：每次訓練的批次大小。在 16GB 記憶體的設備上，若發生記憶體不足（OOM），可考慮調降為 1。
- `--steps-per-report`：每隔 10 個 step 在日誌中輸出一份損失值（Loss）報告。
- `--steps-per-eval`：每隔 200 個 step 進行一次驗證集評估（Validation），計算 Val Loss。
- `--save-every`：每隔 200 個 step 儲存一次模型權重 (Checkpoint)。
- `--learning-rate`：設定學習率為 $10^{-5}$，這是微調常見的穩定數值。
- `--adapter-path`：訓練完成的 LoRA 權重 (Adapter) 儲存目錄。
- `> training_log.txt 2>&1 &`：將正常輸出與錯誤訊息合併導向至 `training_log.txt`，並在背景執行。

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
  --model mlx-community/gemma-4-e2b-it-4bit \
  --adapter-path ./adapters_output \
  --prompt "<start_of_turn>user
你是一位親切、專業的電商導購助手，會根據用戶的需求給出實用的商品建議，並以問句結尾來引導對話。

高CP值的nvidia顯卡<end_of_turn>
<start_of_turn>model
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
| Step 1 | 呼叫 Gemini API 生成最多 50 筆新對話,經過 (a) batch 內三道過濾(過短/重複/異常字元) (b) 與 master 兩階段去重 (c) Python 機械評級,**只有 A/B 級才** append 到 `master_conversations.jsonl`,同時覆蓋寫入 `latest_generated_conversations.jsonl` |
| Step 2 | 重新產生 `train.jsonl` / `valid.jsonl`(合併種子 + master,單輪做同義詞增強) |
| Step 3 | 備份當前 `adapters.safetensors` 為 `.bak`(rollback fallback) |
| Step 4 | `caffeinate -dims` 包裹的 LoRA 訓練 600 iters,`MLX_MAX_BATCH_SIZE=16` |
| Step 5 | 解析 `training_log.txt` 本次 run 的 Val loss,若 `LAST - BEST > OVERFIT_THRESHOLD` (0.15) 則 cp 最佳 checkpoint(`{BEST_ITER:07d}_adapters.safetensors`)覆蓋 `adapters.safetensors`;找不到對應 checkpoint 時改還原 `.bak` 備份 |

---

## 5. 重要路徑

| 路徑 | 說明 |
|------|------|
| `adapters_output/adapters.safetensors` | 當前使用的 LoRA 權重 |
| `adapters_output/adapters.safetensors.bak` | 訓練前備份（過擬合 fallback rollback 用,訓練完成後會清除） |
| `adapters_output/{iter:07d}_adapters.safetensors` | 各 iter 的 checkpoint |
| `adapters_output/adapter_config.json` | LoRA 設定 |
| `data/master_conversations.jsonl` | Gemini 累積生成的對話池(append-only) |
| `data/latest_generated_conversations.jsonl` | 僅本次 Gemini 執行接受的對話(每次覆蓋) |
| `training_log.txt` | 訓練 log(`mlx_lm lora` 輸出,`daily_workflow.sh` 用其末段解析過擬合) |
| `daily_workflow.log` | 每日自動化腳本的執行紀錄 |
