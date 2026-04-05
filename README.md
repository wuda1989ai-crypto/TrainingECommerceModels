# 說明筆記

## 架設訓練模型
安裝 Conda (推薦 Miniforge)
```zsh
brew install miniforge
```

建立一個名為 mlx_env 的環境，並指定 Python 3.11：
```zsh
# 建立環境
conda create -n mlx_env python=3.11 -y

# 使用zsh
conda init zsh

source ~/.zshrc

# 啟用環境 - CLI都要先執行這個指令
conda activate mlx_env
```

```zsh
# 安裝 MLX 專用模型庫
pip install mlx-lm

# 安裝用於轉換 Core ML 的工具 (iOS 開發必備)
pip install coremltools

# 安裝資料處理常用工具
pip install pandas numpy
```

## 執行python產資料
```zsh
python3 generate_ecommerce_data.py
```

## 執行模型訓練
```zsh
nohup python -m mlx_lm.lora \
--model mlx-community/Llama-3.2-3B-Instruct-4bit \
--train \
--data ./data \
--iters 500 \
--batch-size 2 \
--steps-per-report 10 \
--learning-rate 1e-5 \
--adapter-path ./adapters_output > training_log.txt 2>&1 &

```
- 防止系統睡眠-目前沒用
終端機輸入 caffeinate 指令（這能強迫 Mac 保持清醒）：
```zsh
# 在執行訓練指令前先下這行，或另開一個視窗執行
caffeinate -i
```


## 測試
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

## 監看 目前訓練進度
```zsh
tail -f training_log.txt
```


