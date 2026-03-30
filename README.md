# 說明筆記

## 執行
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

## 測試
```zsh
python -m mlx_lm.generate \
--model mlx-community/Llama-3.2-3B-Instruct-4bit \
--adapter-path ./adapters_output \
--prompt "<|user|>\n我想找適合夏天穿的慢跑鞋，預算 2000 元左右。<|assistant|>\n" \
--max-tokens 200
```




