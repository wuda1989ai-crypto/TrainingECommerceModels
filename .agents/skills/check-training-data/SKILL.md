---
name: check-training-data
description: "檢查 mlx-lm 訓練資料品質：格式驗證、內容分析、潛在問題偵測。用於訓練前的資料健檢。"
argument-hint: "[jsonl-file-path]"
allowed-tools: Read Bash Grep Glob
---

# 訓練資料品質檢查

你是一位 MLX LoRA 微調的資料品質審查員。請對指定的 JSONL 訓練資料進行全面檢查。

## 檢查目標

對 `$ARGUMENTS` 指定的檔案進行檢查。若未指定檔案，則依序檢查：
1. `data/train.jsonl`（訓練集）
2. `data/valid.jsonl`（驗證集）
3. `data/latest_generated_conversations.jsonl`（最新生成）
4. `data/master_conversations.jsonl`（主資料庫）

找到第一個存在的檔案即開始檢查。

## 檢查項目

請用 Bash 執行 Python 腳本完成以下所有檢查，最後以表格形式輸出報告。

### 1. 格式驗證
- 每行是否為合法 JSON
- 是否符合 `mlx-lm` 所需的 `{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}` 格式
- 若檔案使用舊格式 `{"user":"...","assistant":"..."}` 或 `{"turns":[...]}` 等非 messages 格式，標記為 **格式不相容**，並提示需要轉換
- 每筆 messages 是否包含 system / user / assistant 三種 role
- 是否有空白的 content 欄位

### 2. 內容統計
- 總筆數
- user 訊息長度：最短、最長、平均（字元數）
- assistant 回覆長度：最短、最長、平均（字元數）
- 多輪對話筆數（messages 中有 2 組以上 user+assistant 的）
- 單輪對話筆數

### 3. 多樣性分析
- assistant 開頭用語分布（統計前 5 個最常見的開頭 pattern，取前 4 個字）
- assistant 結尾類型（問句 vs 非問句比例）
- user 訊息風格分布：
  - 短關鍵字（< 10 字，無標點）
  - 情境式句子（10~40 字）
  - 長描述（> 40 字）
- 商品類別分布（用關鍵字粗估：3C、家電、美妝、服飾、食品、寵物、運動、其他）

### 4. 潛在問題偵測
- 重複的 user 訊息（完整重複 + 正規化後重複）
- assistant 回覆過短（< 30 字）的筆數
- user 訊息過短（< 3 字）的筆數
- 開頭用語過度集中（某個 pattern > 40% 即警告）
- 結尾問句比例過高（> 85% 即警告）
- 類別分布過於集中（某類 > 30% 即警告）

## 輸出格式

請以下列結構輸出報告：

```
## 訓練資料品質報告

檔案：{filename}
檢查時間：{timestamp}

### 格式檢查
| 項目 | 結果 |
|------|------|
| ... | ... |

### 內容統計
| 指標 | 數值 |
|------|------|
| ... | ... |

### 多樣性分析
（各項分析結果）

### 潛在問題
- [PASS/WARN/FAIL] 問題描述

### 總結
（一段話總結資料品質，給出 A/B/C/D 評級，並列出建議改善方向）
```

評級標準：
- **A**：格式正確 + 無 WARN/FAIL
- **B**：格式正確 + 僅有 WARN
- **C**：格式不相容，或有 1 個 FAIL
- **D**：多個 FAIL，資料不適合直接訓練
