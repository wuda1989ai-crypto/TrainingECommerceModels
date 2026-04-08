# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

- 都用繁體中文回答

## 專案目的

使用 Apple 的 `mlx-lm` 框架,在 Apple Silicon 上對 `mlx-community/gemma-4-e2b-it-4bit` 進行 LoRA 微調,目標是打造一個繁體中文的電商導購助手。

## 開發環境

Conda 環境 `mlx_env`(Python 3.11),需安裝 `mlx-lm`、`coremltools`、`pandas`、`numpy`。執行任何指令前請先 `conda activate mlx_env`。

## 常用指令

重新產生訓練/驗證資料(會寫入 `data/train.jsonl` 與 `data/valid.jsonl`):
```zsh
python3 generate_ecommerce_data.py
```

背景執行 LoRA 訓練(log 寫到 `training_log.txt`):
```zsh
nohup python -m mlx_lm.lora \
  --model mlx-community/gemma-4-e2b-it-4bit \
  --train --data ./data \
  --iters 500 --batch-size 2 --steps-per-report 10 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters_output > training_log.txt 2>&1 &
```

監看訓練進度:`tail -f training_log.txt`(注意觀察 `Val loss` 是否過擬合)。

使用訓練後的 adapter 進行推理 — prompt 必須使用完整的 Llama-3 chat template,包含 `<|begin_of_text|>`、`<|start_header_id|>system|user|assistant<|end_header_id|>`、`<|eot_id|>` 等特殊 token(完整範例見 README.md):
```zsh
python -m mlx_lm.generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path ./adapters_output \
  --prompt "<完整 chat-template prompt>" --max-tokens 200
```

長時間訓練時可在另一個終端機執行 `caffeinate -i` 防止系統睡眠。

## 程式架構

資料流由兩個檔案組成,兩者需保持同步:

- **`ecommerce_data.py`** — 純資料模組。匯出 `raw_conversations`(`(user_msg, assistant_msg)` tuple 的清單,為手寫的種子對話)與 `SYNONYM_DICT`(繁中詞語 → 同義詞清單,供資料增強使用)。要新增訓練情境時,請修改這兩個常數,**不要**寫在 generator 裡。

- **`generate_ecommerce_data.py`** — 讀取上述資料並輸出 JSONL。流程如下:
  1. 打亂 `raw_conversations`,約 20% 切為驗證集、約 80% 為訓練集。
  2. 每筆訓練樣本除了原始句子外,會透過 `augment_sentence_with_synonyms_chinese` 再產生最多 3 筆增強版本;該函式會從 `SYNONYM_DICT` 中隨機挑一個命中的詞語替換,**只替換 user 訊息**,assistant 回答絕不更動。
  3. 打亂訓練集,寫入 `data/train.jsonl` 與 `data/valid.jsonl`。
  4. 驗證集**永遠不做增強** — 保留原始樣本才能得到可信的 `Val loss`。

**輸出格式為 `messages` 樣式的 JSONL**(非 ChatML 的 `text` 格式),每行一筆:
```json
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```
`mlx-lm` 會從此結構自動套用 Llama-3 的 chat template。`system_prompt` 是定義在 `generate_ecommerce_dataset()` 裡的字串常數,要改請直接修改該處。注意:`GEMINI.md` 描述的是舊版 `{"text": ...}` ChatML 格式,實際程式碼輸出的是 `messages` 格式,**以程式碼為準**。

訓練產物會存放在 `adapters_output/`(LoRA `.safetensors` 權重與設定檔)。推理時必須同時提供 `--model`(基礎模型)與 `--adapter-path`,請勿搬動或重新命名此資料夾,否則先前的 checkpoint 將無法載入。
