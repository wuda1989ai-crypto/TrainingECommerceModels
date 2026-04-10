#
#  generate_ecommerce_data.py
#  TrainingECommerceModels
#
#  Created by Wuda on 2026/3/31.
#

import json
import os
import random
from ecommerce_data import SYNONYM_DICT, raw_conversations, multi_turn_conversations

MASTER_FILE = "data/master_conversations.jsonl"


def load_master_conversations():
    """
    讀取 data/master_conversations.jsonl (由 generate_gemini_data.py 產出),
    回傳 (single_turn, multi_turn):
      single_turn — list[(user, assistant)]，與 ecommerce_data.raw_conversations 同構
      multi_turn  — list[list[(role, content)]]，與 ecommerce_data.multi_turn_conversations 同構
    檔案不存在時回傳兩個空 list。
    """
    if not os.path.exists(MASTER_FILE):
        return [], []
    single_turn = []
    multi_turn = []
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "turns" in obj and isinstance(obj["turns"], list):
                    # 多輪對話: 轉換為 [("user", "..."), ("assistant", "..."), ...] 格式
                    turns = []
                    for turn in obj["turns"]:
                        u = turn.get("user", "").strip()
                        a = turn.get("assistant", "").strip()
                        if u and a:
                            turns.append(("user", u))
                            turns.append(("assistant", a))
                    if len(turns) >= 4:  # 至少 2 輪 (2 組 user+assistant)
                        multi_turn.append(turns)
                else:
                    user_msg = obj.get("user", "").strip()
                    assistant_msg = obj.get("assistant", "").strip()
                    if user_msg and assistant_msg:
                        single_turn.append((user_msg, assistant_msg))
            except (json.JSONDecodeError, KeyError):
                continue
    return single_turn, multi_turn


def generate_ecommerce_dataset():
    if not os.path.exists("data"):
        os.makedirs("data")

    # 使用 messages 陣列格式，MLX-LM 會自動套用 Llama-3 的 Chat Template
    system_prompt = "你是一位親切、專業的電商導購助手，會根據用戶的需求給出實用的商品建議。視情況以問句追問細節、或以總結語與行動建議收尾，讓對話自然流暢。"

    # --- 資料增強函數 ---
    def augment_sentence_with_synonyms_chinese(sentence, synonym_dict, num_augmentations=1):
        augmented_sentences = [sentence]
        
        # 找出句子中所有可以替換的詞語或短語
        possible_replacements = []
        for original_phrase, syn_list in synonym_dict.items():
            if original_phrase in sentence and syn_list:
                possible_replacements.append((original_phrase, syn_list))

        if not possible_replacements: # 如果沒有可替換的詞語，則直接返回原始句子
            return augmented_sentences

        for _ in range(num_augmentations):
            temp_sentence = sentence
            # 每次增強隨機選擇一個詞語或短語進行替換
            phrase_to_replace, syn_list = random.choice(possible_replacements)
            
            chosen_synonym = random.choice(syn_list)
            # 替換句子中所有出現的該詞語或短語
            temp_sentence = temp_sentence.replace(phrase_to_replace, chosen_synonym)
            
            # 只有當句子實際發生變化且是新的增強結果時才加入
            if temp_sentence != sentence and temp_sentence not in augmented_sentences:
                augmented_sentences.append(temp_sentence)
        
        return augmented_sentences
    # --- 資料增強函數結束 ---

    # 1. 將原始資料集分為訓練集和驗證集
    # 來源 = ecommerce_data.raw_conversations (種子) + master_conversations.jsonl (Gemini 堆疊)
    # 複製一份 raw_conversations,避免污染 import 進來的模組層級變數
    master_single, master_multi = load_master_conversations()
    all_conversations = list(raw_conversations) + master_single
    print(f"📚 資料來源: 種子 {len(raw_conversations)} 筆 + master 單輪 {len(master_single)} 筆 = {len(all_conversations)} 筆")
    print(f"  ↳ master 多輪對話: {len(master_multi)} 筆")

    random.shuffle(all_conversations) # 打亂原始對話，確保隨機分割
    # 通常會將約 10-20% 的資料用於驗證。這裡使用約 20%。
    num_validation_samples = max(1, len(all_conversations) // 5)

    valid_raw_conversations = all_conversations[:num_validation_samples]
    train_raw_conversations = all_conversations[num_validation_samples:]

    # 2. 處理訓練集：加入原始樣本並進行資料增強
    full_dataset = []
    for user_msg, assistant_msg in train_raw_conversations:
        # 加入原始樣本
        full_dataset.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        })

        # 對用戶訊息進行資料增強 (增加增強次數以增加樣態)
        augmented_user_msgs = augment_sentence_with_synonyms_chinese(user_msg, SYNONYM_DICT, num_augmentations=3)
        for aug_user_msg in augmented_user_msgs:
            if aug_user_msg != user_msg: # 避免重複加入原始訊息
                full_dataset.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": aug_user_msg},
                        {"role": "assistant", "content": assistant_msg} # 助手的回答保持不變
                    ]
                })
    
    random.shuffle(full_dataset) # 打亂增強後的訓練集，增加訓練的隨機性

    # 3. 處理驗證集：只使用原始樣本，不進行增強
    validation_dataset = []
    for user_msg, assistant_msg in valid_raw_conversations:
        validation_dataset.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
        })

    # 4. 加入多輪對話：切分 train/valid，不做增強（接續語境難以增強）
    # 合併種子多輪 + master 多輪
    shuffled_multi = list(multi_turn_conversations) + master_multi
    random.shuffle(shuffled_multi)
    num_multi_valid = max(1, len(shuffled_multi) // 5)
    multi_valid = shuffled_multi[:num_multi_valid]
    multi_train = shuffled_multi[num_multi_valid:]

    for turns in multi_train:
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": role, "content": content} for role, content in turns]
        full_dataset.append({"messages": messages})

    for turns in multi_valid:
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": role, "content": content} for role, content in turns]
        validation_dataset.append({"messages": messages})

    random.shuffle(full_dataset)
    print(f"  ↳ 多輪對話: 訓練 {len(multi_train)} 筆 / 驗證 {len(multi_valid)} 筆")

    with open("data/train.jsonl", "w", encoding="utf-8") as file:
        for item in full_dataset:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open("data/valid.jsonl", "w", encoding="utf-8") as file:
        for item in validation_dataset:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully generated train.jsonl ({len(full_dataset)} samples) and valid.jsonl ({len(validation_dataset)} samples).")

if __name__ == "__main__":
    generate_ecommerce_dataset()
