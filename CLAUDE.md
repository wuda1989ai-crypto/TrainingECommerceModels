# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

- 都用繁體中文回答

## 專案目的

使用 Apple 的 `mlx-lm` 框架,在 Apple Silicon 上對 `mlx-community/gemma-4-e2b-it-4bit` 進行 LoRA 微調,目標是打造一個繁體中文的電商導購助手。

## 開發環境

Conda 環境 `mlx_env`(Python 3.11),需安裝 `mlx-lm`、`coremltools`、`pandas`、`numpy`、`google-genai`(自動化資料生成用)。執行任何指令前請先 `conda activate mlx_env`。

## 常用指令

呼叫 Gemini 生成新對話並 append 到 `data/master_conversations.jsonl`(需先設定 `GEMINI_API_KEY`):
```zsh
python3 generate_gemini_data.py
```

重新產生訓練/驗證資料(會寫入 `data/train.jsonl` 與 `data/valid.jsonl`):
```zsh
python3 generate_ecommerce_data.py
```

背景執行 LoRA 訓練(log 寫到 `training_log.txt`,參數需與 `daily_workflow.sh` 對齊):
```zsh
export MLX_MAX_BATCH_SIZE=16 && nohup python -m mlx_lm lora \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --train --data ./data \
  --iters 600 --batch-size 1 --steps-per-report 10 \
  --steps-per-eval 200 --save-every 200 \
  --learning-rate 1e-5 --max-seq-length 512 \
  --adapter-path ./adapters_output > training_log.txt 2>&1 &
```

執行每日全自動流程(資料生成 → 重建資料集 → 備份 adapter → 訓練 → 過擬合回捲):
```zsh
./daily_workflow.sh 2>&1 | tee daily_workflow.log
```

監看訓練進度:`tail -f training_log.txt`(注意觀察 `Val loss` 是否過擬合)。

使用訓練後的 adapter 進行推理 — Gemma 使用 `<start_of_turn>user / <start_of_turn>model` 標記的 chat template,完整範例見 README.md:
```zsh
python -m mlx_lm.generate \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --adapter-path ./adapters_output \
  --prompt "<完整 chat-template prompt>" --max-tokens 200
```

長時間訓練時可在另一個終端機執行 `caffeinate -i` 防止系統睡眠;`daily_workflow.sh` 已用 `caffeinate -dims` 包裹訓練指令。

## 程式架構

資料流由四個檔案組成,層次為「種子資料 + Gemini 增量 → 訓練資料」:

- **`ecommerce_data.py`** — 純資料模組。匯出 `raw_conversations`(`(user_msg, assistant_msg)` tuple 的清單,為手寫的種子對話)、`multi_turn_conversations`(多輪對話清單)與 `SYNONYM_DICT`(繁中詞語 → 同義詞清單,供資料增強使用)。要新增手寫訓練情境時,請修改這三個常數,**不要**寫在 generator 裡。

- **`generate_gemini_data.py`** — Gemini API 自動生成器。流程:
  1. 從 `CATEGORY_POOL`(12 個主分類)隨機抽 4 主類 × 8 子項組成 prompt 區塊,強迫跨 batch 類別分布平均。
  2. 呼叫 Gemini(預設 `gemini-3.1-flash-lite-preview`)並在 prompt 中精確指定風格/字數/結尾類型/類別配額,每 batch 預設 25 筆,最多 `MAX_CALLS=8` 次呼叫,目標 `TARGET_COUNT=50` 筆。
  3. **三道後處理**(任一違規整筆丟棄):過短 user(< 3 字)、batch 內正規化重複、異常字元(白名單僅允許繁中 + 英數 + 常見標點 + emoji,攔截簡中/日韓假名/印度語系/程式碼殘渣)。
  4. **兩階段去重**比對 `master_conversations.jsonl`:精確字串 + NFKC 正規化 SHA1 hash。多輪對話以第一輪 user 為去重基準。
  5. **Python 機械化品質審查**(`evaluate_data_quality`),統計指標包含字數分桶(短 < 10 / 中 10–40 / 長 > 40)、結尾問句比例、開頭用語集中度、商品類別分布等,評為 A/B/C/D 四級。**只有 A 或 B 級才會寫入 master**,C/D 級直接 `sys.exit(1)` 放棄,避免污染資料集。
  6. 通過後 append 到 `data/master_conversations.jsonl`(累加),同時覆蓋寫入 `data/latest_generated_conversations.jsonl`(僅本次結果,方便檢視)。
  - 環境變數 `GEMINI_API_KEY` 必填。要切換模型只需改檔案頂部的 `MODEL_NAME` 常數。

- **`generate_ecommerce_data.py`** — 讀取種子 + master 並輸出訓練 JSONL。流程:
  1. 從 `ecommerce_data.py` 載入 `raw_conversations` / `multi_turn_conversations`,並讀取 `data/master_conversations.jsonl`(由 Gemini 累積),合併成單輪與多輪兩個池。
  2. 打亂後切分,約 20% 為驗證集、約 80% 為訓練集。
  3. 每筆**單輪**訓練樣本除原句外,會透過 `augment_sentence_with_synonyms_chinese` 再產生最多 3 筆增強版本;只替換 user 訊息,assistant 回答絕不更動。多輪對話**不做同義詞增強**。
  4. 打亂訓練集,寫入 `data/train.jsonl` 與 `data/valid.jsonl`。
  5. 驗證集**永遠不做增強** — 保留原始樣本才能得到可信的 `Val loss`。

- **`daily_workflow.sh`** — 每日自動化訓練腳本。流程:Gemini 生成 → 重建資料集 → 備份當前 `adapters.safetensors` → `caffeinate -dims` 包裹的 LoRA 訓練(`ITERS=600`, `--steps-per-eval 200`, `--save-every 200`,兩者必須對齊以保證最佳 Val loss 對應的 iter 有 checkpoint)→ **過擬合自動回捲**:解析本次 run 的 Val loss,若 `LAST - BEST > OVERFIT_THRESHOLD`(預設 0.15)則 cp `{BEST_ITER:07d}_adapters.safetensors` 覆蓋 `adapters.safetensors`;找不到對應 checkpoint 時 fallback 還原訓練前的 `.bak` 備份。腳本以 `set -euo pipefail` 啟動,會自動 `conda activate mlx_env`。

**輸出格式為 `messages` 樣式的 JSONL**(非 ChatML 的 `text` 格式),每行一筆:
```json
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```
`mlx-lm` 會從此結構自動套用基座模型(Gemma 4)的 chat template。`system_prompt` 是定義在 `generate_ecommerce_dataset()` 裡的字串常數,要改請直接修改該處。

訓練產物會存放在 `adapters_output/`(LoRA `.safetensors` 權重與設定檔)。推理時必須同時提供 `--model`(基礎模型)與 `--adapter-path`,請勿搬動或重新命名此資料夾,否則先前的 checkpoint 將無法載入。
