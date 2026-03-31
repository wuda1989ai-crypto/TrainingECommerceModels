#
#  generate_ecommerce_data.py
#  TrainingECommerceModels
#
#  Created by Wuda on 2026/3/31.
#

import json
import os
import random
from ecommerce_data import SYNONYM_DICT, raw_conversations

def generate_ecommerce_dataset():
    if not os.path.exists("data"):
        os.makedirs("data")

    # 使用 messages 陣列格式，MLX-LM 會自動套用 Llama-3 的 Chat Template
    system_prompt = "你是一位親切、專業的電商導購助手，會根據用戶的需求給出實用的商品建議，並以問句結尾來引導對話。"

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
    random.shuffle(raw_conversations) # 打亂原始對話，確保隨機分割
    # 通常會將約 10-20% 的資料用於驗證。這裡使用約 20%。
    num_validation_samples = max(1, len(raw_conversations) // 5) 
    
    valid_raw_conversations = raw_conversations[:num_validation_samples]
    train_raw_conversations = raw_conversations[num_validation_samples:]

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

    with open("data/train.jsonl", "w", encoding="utf-8") as file:
        for item in full_dataset:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open("data/valid.jsonl", "w", encoding="utf-8") as file:
        for item in validation_dataset:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully generated train.jsonl ({len(full_dataset)} samples) and valid.jsonl ({len(validation_dataset)} samples).")

if __name__ == "__main__":
    generate_ecommerce_dataset()
