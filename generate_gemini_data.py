#
#  generate_gemini_data.py
#  TrainingECommerceModels
#
#  每日呼叫 Gemini 2.5 Pro 生成 1000 筆電商導購對話,
#  以 tuple 格式 (user, assistant) append 到 data/master_conversations.jsonl。
#
#  去重策略 (B + C 組合):
#    B. 精確字串: user 訊息與 master 既有完全相同 → 丟棄
#    C. 正規化 hash: 去空白、全形轉半形、去常見標點、小寫後做 SHA1 → 丟棄
#
#  環境變數:
#    GEMINI_API_KEY  — 必填
#

import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("❌ 請先安裝新版 SDK: pip install -q -U google-genai")


# ---- 基本設定 ----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MASTER_FILE = PROJECT_DIR / "data" / "master_conversations.jsonl"
MODEL_NAME = "gemini-3-flash-preview"

TARGET_COUNT = 50          # 每日目標新增筆數
BATCH_SIZE = 25              # 每次 API 呼叫要幾筆 (控制輸出 token)
MAX_CALLS = 5               # 安全上限,避免去重後補不滿無限呼叫
RETRY_PER_CALL = 2           # 單一 batch 失敗重試次數
SLEEP_BETWEEN_CALLS = 1.0    # 每次呼叫間隔 (秒),友善 rate limit

# ---- Gemini 系統提示 ---------------------------------------------------------
# 為了讓訓練資料貼近「真實電商流量」的樣態,此 prompt 強制 Gemini 在單一 batch
# 內產出多種風格的 user query (關鍵字式、情境式、比較式、規格式、長描述、售後),
# 並嚴格要求 batch 內 user 長度與句式必須有明顯差異,降低語意重複率。
GEMINI_SYSTEM_PROMPT = """你是一位資料集生成助手,專門為繁體中文電商導購模型產生訓練對話。

請生成 {n} 筆繁體中文的電商導購對話,每筆對話包含 "user" 與 "assistant" 兩個欄位。

# 【重要】user query 風格分布 — 這 {n} 筆「必須」依下列比例混合,不能全部寫成同一種風格:

1. **情境式完整句子 (約 30%)** — 長度 20~40 字,帶使用者身分與情境。
   例: 「我是租屋族,房間小,想要一台不佔空間的吸塵器」
       「下個月要帶爸媽去日本,想買雙好走的運動鞋」

2. **短關鍵字搜尋 (約 25%)** — 長度 3~10 字,**完全沒有語氣詞、動詞、標點**,純粹像在 Google 或電商 App 搜尋框打字。
   例: 「高cp值 nvidia 顯卡」「無線耳機推薦」「便宜吸塵器」
       「送禮 長輩 實用」「iphone 15 pro 256 黑」「dyson v12 保養」
   ⚠️ 這類請**一定要包含**,這是大眾最常見的搜尋行為。

3. **品牌/型號比較 (約 15%)** — 直接詢問 A 跟 B 哪個好、XXX 值不值得買。
   例: 「Dyson V12 跟 V15 差在哪?」
       「Sony WF-1000XM5 跟 Bose QC Ultra 哪個抗噪比較強?」
       「MacBook Air M3 值得買嗎還是等 M4?」

4. **具體規格/型號詢問 (約 15%)** — 明確帶型號、容量、顏色、版本號。
   例: 「iPhone 16 Pro Max 256G 黑鈦跟白鈦哪個比較耐看?」
       「PS5 Slim 光碟版現在還買得到嗎?」
       「Switch 2 首發版跟普通版差多少?」

5. **超長多條件需求 (約 10%)** — 長度 60 字以上,一次列出多個限制條件。
   例: 「我家客廳 15 坪,有兩個小孩會亂跑,想買吸塵器但預算不能超過五千,希望吸力強還要能吸床墊塵蟎,品牌保固也要久一點,請幫我推薦」

6. **售後/使用問題 (約 5%)** — 已購後的疑問、保固、維修、使用困擾。
   例: 「藍牙耳機只有一邊有聲音,保固怎麼處理?」
       「昨天下單今天就降價,可以退差價嗎?」
       「Dyson 濾網多久要換一次?」

# 商品類別要廣泛分散,涵蓋:
3C、家電、廚房用品、運動用品、美妝保養、服飾、文具、寵物、嬰幼兒、戶外、汽機車周邊、保健、傢俱、圖書、禮品 等。**同一個 batch 內,類別不可集中在同 3 類**。

# assistant 回覆規範:
1. 以親切、專業的電商導購助手口吻回答。
2. 給出 2~4 個具體商品建議或選購重點,包含簡短理由 (價格帶、適用情境、材質、品牌特色)。
3. **必須以問句結尾**,引導用戶進一步對話 (例如「請問您的預算大約多少?」)。
4. 回答長度 80~200 字,避免流水帳。
5. 針對風格 2 (短關鍵字) 與 6 (售後) 的 user,assistant 也要合理應對 — 短關鍵字就猜測用戶意圖並給建議,售後就給處理流程並問細節,不要每一筆都假裝用戶給了完整情境。

# 【去重要求】
- 本 batch 的 {n} 筆之間,user query 的**主題、句式、長度必須明顯不同**。
- 同一個商品類別在一個 batch 內不要出現超過 2 次。
- 嚴禁產出「只改幾個字就幾乎一樣」的近義改寫 (例如「租屋族想要不佔空間的吸塵器」和「小套房想找省空間的吸塵器」)。

# 輸出格式
請嚴格回傳 JSON 陣列,**不要**有任何前後說明文字、markdown code block:
[
  {{"user": "...", "assistant": "..."}},
  ...
]
"""


# ---- 正規化 (方案 C) ---------------------------------------------------------
_PUNCT_PATTERN = re.compile(
    r"[\s、,。.!?！？…~～\-—_/\\\"'`“”‘’「」『』()（）\[\]【】{}《》<>:;：;·]+"
)


def normalize_for_dedupe(text: str) -> str:
    """
    去除空白、全形轉半形、去常見標點、小寫化。
    用於方案 C 的正規化 hash 去重。
    """
    if not text:
        return ""
    # NFKC 會把全形英數/符號轉成半形
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_PATTERN.sub("", text)
    return text.strip()


def hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---- Master 檔讀寫 ------------------------------------------------------------
def load_master_dedupe_sets():
    """
    讀取既有 master_conversations.jsonl,回傳:
      exact_set      — 完整 user 字串集合 (方案 B)
      normalized_set — 正規化後 hash 集合 (方案 C)
    """
    exact_set = set()
    normalized_set = set()
    if not MASTER_FILE.exists():
        return exact_set, normalized_set

    with MASTER_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                user_msg = obj.get("user", "")
                exact_set.add(user_msg)
                normalized_set.add(hash_key(normalize_for_dedupe(user_msg)))
            except json.JSONDecodeError:
                continue
    return exact_set, normalized_set


def append_to_master(pairs):
    """以 tuple 格式 {"user","assistant"} 逐行 append 到 master 檔。"""
    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_FILE.open("a", encoding="utf-8") as f:
        for user_msg, assistant_msg in pairs:
            obj = {"user": user_msg, "assistant": assistant_msg}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---- Gemini 呼叫 -------------------------------------------------------------
def call_gemini_batch(client, n: int):
    """
    呼叫 Gemini 產生 n 筆對話,回傳 list[(user, assistant)]。
    response 解析失敗會 raise,由外層重試。
    """
    prompt = GEMINI_SYSTEM_PROMPT.format(n=n)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.95,
            top_p=0.95,
            response_mime_type="application/json",
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini 回傳空字串")
    data = json.loads(text)  # 失敗就讓外層 retry
    if not isinstance(data, list):
        raise ValueError("Gemini 回傳非 list")

    pairs = []
    for item in data:
        user_msg = (item.get("user") or "").strip()
        assistant_msg = (item.get("assistant") or "").strip()
        if user_msg and assistant_msg:
            pairs.append((user_msg, assistant_msg))
    return pairs


# ---- 主流程 ------------------------------------------------------------------
def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("❌ 未設定環境變數 GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    exact_set, normalized_set = load_master_dedupe_sets()
    print(f"📚 既有 master 對話筆數(dedupe 基準): {len(exact_set)}")
    print(
        f"🤖 Model: {MODEL_NAME} | batch={BATCH_SIZE} | target={TARGET_COUNT}"
    )
    print("   提醒: Flash 類模型單次呼叫通常為數秒到數十秒,請依實際輸出觀察耗時。")

    accepted = []           # 本次 run 接受的新對話
    total_calls = 0
    total_from_api = 0
    total_rejected = 0

    while len(accepted) < TARGET_COUNT and total_calls < MAX_CALLS:
        total_calls += 1
        remaining = TARGET_COUNT - len(accepted)
        batch_n = min(BATCH_SIZE, remaining)

        # 單次呼叫重試
        pairs = None
        for attempt in range(RETRY_PER_CALL + 1):
            print(
                f"  → call #{total_calls} attempt {attempt + 1}: "
                f"向 Gemini 請求 {batch_n} 筆 (thinking 中,請稍候)...",
                flush=True,
            )
            t0 = time.time()
            try:
                pairs = call_gemini_batch(client, batch_n)
                elapsed = time.time() - t0
                print(
                    f"    ← 回應完成, 耗時 {elapsed:.1f}s, 解析得 {len(pairs)} 筆",
                    flush=True,
                )
                break
            except Exception as e:
                elapsed = time.time() - t0
                print(
                    f"  ⚠️  call #{total_calls} attempt {attempt + 1} "
                    f"失敗 ({elapsed:.1f}s): {e}",
                    flush=True,
                )
                time.sleep(2 * (attempt + 1))
        if not pairs:
            print(f"  ❌ call #{total_calls} 連續失敗,跳過此 batch")
            continue

        total_from_api += len(pairs)

        # 去重 (B + C + 本次 run 內部)
        batch_accepted = 0
        for user_msg, assistant_msg in pairs:
            if user_msg in exact_set:
                total_rejected += 1
                continue
            norm_hash = hash_key(normalize_for_dedupe(user_msg))
            if norm_hash in normalized_set:
                total_rejected += 1
                continue

            exact_set.add(user_msg)
            normalized_set.add(norm_hash)
            accepted.append((user_msg, assistant_msg))
            batch_accepted += 1
            if len(accepted) >= TARGET_COUNT:
                break

        print(
            f"  ✓ call #{total_calls}: API 回傳 {len(pairs)},"
            f" 接受 {batch_accepted}, 累計 {len(accepted)}/{TARGET_COUNT}"
        )
        time.sleep(SLEEP_BETWEEN_CALLS)

    # 寫入 master
    if accepted:
        append_to_master(accepted)

    print("─" * 60)
    print(f"🎯 完成: 本次新增 {len(accepted)} 筆到 {MASTER_FILE}")
    print(f"   API 呼叫次數: {total_calls}")
    print(f"   API 回傳總筆數: {total_from_api}")
    print(f"   被去重拒絕筆數: {total_rejected}")
    if len(accepted) < TARGET_COUNT:
        print(f"⚠️  未達目標 {TARGET_COUNT} 筆,可能是 API 失敗次數過多或去重過嚴")


if __name__ == "__main__":
    main()
