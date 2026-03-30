# TrainingECommerceModels 開發指南

本專案旨在為電子商務場景微調大型語言模型（LLM），使其具備專業導購與客戶服務的能力。專案採用 Apple Silicon 優化的 `mlx-lm` 框架。

## 專案概述
- **基礎模型**: `mlx-community/Llama-3.2-3B-Instruct-4bit`
- **技術路徑**: 使用 LoRA (Low-Rank Adaptation) 技術進行微調。
- **應用場景**: 包含夏季鞋款推薦、長輩禮物建議、戶外服飾挑選等電商導購對話。

## 目錄結構
- `data/`: 存放訓練用的 `train.jsonl` 與驗證用的 `valid.jsonl`。
- `adapters_output/`: 訓練過程中生成的 LoRA 適配器權重（`.safetensors`）與配置。
- `generate_ecommerce_data.py`: 用於生成模擬電商對話資料的工具腳本。
- `training_log.txt`: 記錄訓練過程中的損失值（Loss）與效能指標。

## 核心指令

### 1. 資料準備
修改 `generate_ecommerce_data.py` 中的 `training_samples` 後執行，以更新訓練集：
```bash
python generate_ecommerce_data.py
```

### 2. 模型訓練 (LoRA)
使用 MLX 進行訓練，預設配置為 500 次迭代，每 100 次保存一次權重：
```bash
python -m mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --data ./data \
  --train \
  --iters 500 \
  --save-every 100 \
  --batch-size 2 \
  --adapter-path ./adapters_output
```
*註：建議在 Apple Silicon 設備上執行。*

### 3. 模型推理 (測試)
載入微調後的適配器進行對話測試：
```bash
python -m mlx_lm.generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path ./adapters_output \
  --prompt "<|user|>\n我想找適合夏天穿的慢跑鞋\n<|assistant|>\n"
```

## 開發規範
- **資料格式**: 必須遵循 ChatML 風格的 JSONL 格式：`{"text": "<|user|>\n...\n<|assistant|>\n..."}`。
- **權重管理**: 適配器權重統一存放在 `adapters_output/`，請勿隨意更動目錄結構，以免推理時路徑失效。
- **效能監控**: 訓練時應觀察 `training_log.txt` 中的 `Val loss`，確保模型沒有過擬合（Overfitting）。
