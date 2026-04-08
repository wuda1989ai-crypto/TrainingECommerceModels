#
#  generate_gemini_data.py
#  TrainingECommerceModels
#
#  每日呼叫 Gemini LLM 生成 1000 筆電商導購對話,
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
from enum import Enum
from pathlib import Path

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("❌ 請先安裝新版 SDK: pip install -q -U google-genai")


# ---- 模型列表 ----------------------------------------------------------------
class GeminiModel(str, Enum):
    GEMINI_31_PRO_PREVIEW      = "gemini-3.1-pro-preview"
    GEMINI_3_FLASH_PREVIEW     = "gemini-3-flash-preview"
    GEMINI_31_FLASH_LITE       = "gemini-3.1-flash-lite-preview"
    GEMINI_25_PRO              = "gemini-2.5-pro"
    GEMINI_25_FLASH            = "gemini-2.5-flash"
    GEMINI_25_FLASH_LITE       = "gemini-2.5-flash-lite"


# ---- 基本設定 ----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MASTER_FILE = PROJECT_DIR / "data" / "master_conversations.jsonl"
LATEST_FILE = PROJECT_DIR / "data" / "latest_generated_conversations.jsonl"
MODEL_NAME = GeminiModel.GEMINI_25_FLASH_LITE  # ← 切換模型只需改這一行

TARGET_COUNT = 50          # 要 AI 新增的筆數
BATCH_SIZE = 25            # 每次 API 呼叫要幾筆 (控制輸出 token)
MAX_CALLS = 5               # 安全上限,避免去重後補不滿無限呼叫
RETRY_PER_CALL = 2           # 單一 batch 失敗重試次數
SLEEP_BETWEEN_CALLS = 1.5    # 每次呼叫間隔 (秒),友善 rate limit

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

# 【多輪對話】約 15% 的對話請設計為 2~3 輪往返,模擬真實的追問場景。
格式為 turns 陣列,每個 turn 包含 user 和 assistant:
  {{"turns": [
    {{"user": "筆電推薦 輕薄", "assistant": "輕薄筆電目前熱門的有 MacBook Air M3 和 ASUS Zenbook 14…請問主要用途是?"}},
    {{"user": "主要寫程式跟跑 Docker", "assistant": "如果要跑 Docker,建議 RAM 至少 16GB…"}}
  ]}}
多輪對話的 user 追問應該要自然地延續上一輪的話題,例如補充需求、追問細節、比較選項等。

# 商品類別要廣泛分散,涵蓋:
3C、家電、廚房用品、運動用品、美妝保養、服飾、文具、寵物、嬰幼兒、戶外、汽機車周邊、保健、傢俱、圖書、禮品 等。**同一個 batch 內,類別不可集中在同 3 類**。

# assistant 回覆規範:
1. 以親切、專業的電商導購助手口吻回答。
2. 給出 2~4 個具體商品建議或選購重點,包含簡短理由 (價格帶、適用情境、材質、品牌特色)。
3. **約 60% 以問句結尾**引導進一步對話,**其餘 40% 用總結句、行動建議或祝福語結尾**。嚴禁每一筆都以問句結尾。
4. 回答長度 80~200 字,避免流水帳。
5. 針對風格 2 (短關鍵字) 與 6 (售後) 的 user,assistant 也要合理應對 — 短關鍵字就猜測用戶意圖並給建議,售後就給處理流程並問細節,不要每一筆都假裝用戶給了完整情境。
6. **開頭用語必須多樣化**,嚴禁超過 30% 的回覆以「您好」開頭。請混合使用以下開頭方式:
   - 直接切入主題 (例:「這兩款各有優勢…」「針對您的需求…」)
   - 肯定用戶的選擇 (例:「很棒的選擇！」「這個預算可以挑到不錯的…」)
   - 簡短回應後進入推薦 (例:「沒問題,幫您整理幾個方向。」)
   - 少數才用「您好！」或「嗨！」
7. **只推薦真實存在且知名度高的品牌與型號**。如果不確定某型號是否存在,改用品牌名 + 系列名 (例:「Panasonic 的 NA 系列」),不要自己編造型號。

# 【去重要求】
- 本 batch 的 {n} 筆之間,user query 的**主題、句式、長度必須明顯不同**。
- 同一個商品類別在一個 batch 內不要出現超過 2 次。
- 嚴禁產出「只改幾個字就幾乎一樣」的近義改寫 (例如「租屋族想要不佔空間的吸塵器」和「小套房想找省空間的吸塵器」)。

# 輸出格式
請嚴格回傳 JSON 陣列,**不要**有任何前後說明文字、markdown code block。
單輪對話用 {{"user":"…","assistant":"…"}},多輪對話用 {{"turns":[{{"user":"…","assistant":"…"}},…]}}:
[
  {{"user": "...", "assistant": "..."}},
  {{"turns": [{{"user": "...", "assistant": "..."}}, {{"user": "...", "assistant": "..."}}]}},
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
                if "turns" in obj:
                    user_msg = obj["turns"][0].get("user", "") if obj["turns"] else ""
                else:
                    user_msg = obj.get("user", "")
                if user_msg:
                    exact_set.add(user_msg)
                    normalized_set.add(hash_key(normalize_for_dedupe(user_msg)))
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
    return exact_set, normalized_set


def _serialize_item(item):
    """將單輪 tuple 或多輪 list 序列化為 JSON 字串。"""
    if isinstance(item, list):
        # 多輪: list of (user, assistant) tuples
        turns = [{"user": u, "assistant": a} for u, a in item]
        return json.dumps({"turns": turns}, ensure_ascii=False)
    else:
        # 單輪: (user, assistant) tuple
        user_msg, assistant_msg = item
        return json.dumps({"user": user_msg, "assistant": assistant_msg}, ensure_ascii=False)


def _get_first_user_msg(item):
    """取得用於去重的第一個 user 訊息。"""
    if isinstance(item, list):
        return item[0][0]  # 多輪取第一輪 user
    return item[0]  # 單輪


def append_to_master(pairs):
    """將單輪 tuple 或多輪 list 逐行 append 到 master 檔。"""
    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_FILE.open("a", encoding="utf-8") as f:
        for item in pairs:
            f.write(_serialize_item(item) + "\n")


def write_to_latest(pairs):
    """將本次生成的資料覆蓋寫入 latest 檔，方便確認當次抓取的內容。"""
    LATEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LATEST_FILE.open("w", encoding="utf-8") as f:
        for item in pairs:
            f.write(_serialize_item(item) + "\n")


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
        if "turns" in item and isinstance(item["turns"], list):
            # 多輪對話: 收集為 list of (user, assistant) tuples
            turns = []
            for turn in item["turns"]:
                u = (turn.get("user") or "").strip()
                a = (turn.get("assistant") or "").strip()
                if u and a:
                    turns.append((u, a))
            if len(turns) >= 2:
                pairs.append(turns)  # list of tuples = 多輪
        else:
            user_msg = (item.get("user") or "").strip()
            assistant_msg = (item.get("assistant") or "").strip()
            if user_msg and assistant_msg:
                pairs.append((user_msg, assistant_msg))  # single tuple = 單輪
    return pairs


# ---- 資料審查 ----------------------------------------------------------------
def _serialize_as_messages(item):
    """將 pair 轉為 mlx-lm 所需的 messages 格式 JSON 字串。"""
    system_prompt = "你是一位親切、專業的電商導購助手，會根據用戶的需求給出實用的商品建議，並以問句結尾來引導對話。"
    messages = [{"role": "system", "content": system_prompt}]
    if isinstance(item, list):
        for u, a in item:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
    else:
        messages.append({"role": "user", "content": item[0]})
        messages.append({"role": "assistant", "content": item[1]})
    return json.dumps({"messages": messages}, ensure_ascii=False)


def evaluate_data_quality(client, pairs):
    skill_file = PROJECT_DIR / ".agents" / "skills" / "check-training-data" / "SKILL.md"
    if not skill_file.exists():
        print("⚠️ 找不到 check-training-data skill，跳過檢查")
        return True

    skill_prompt = skill_file.read_text(encoding="utf-8")
    jsonl_str = "\n".join([_serialize_as_messages(item) for item in pairs])
    
    prompt = f"""請扮演資料品質審查員，並遵循以下 Skill 提示內的「檢查項目」、「潛在問題偵測」與「評級標準」，直接閱讀下列 JSONL 內容並進行分析。
請注意：你不必編寫 Python 或 Bash 腳本，直接閱讀資料內容進行評估即可。
在回覆的最後一段，請務必按照格式給出你的最終評級 (A/B/C/D)，例如：「最終評級：A」。

=== Skill 內容 ===
{skill_prompt}

=== 待檢查的資料 ===
{jsonl_str}
"""
    print("\n🧐 正在執行 check-training-data skill 檢查本次資料...", flush=True)
    try:
        response = client.models.generate_content(
            model=GeminiModel.GEMINI_25_FLASH,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
            )
        )
        report = response.text
        
        print("\n" + "="*20 + " 品質檢查報告 " + "="*20)
        print(report)
        print("="*54 + "\n")
        
        import re
        match = re.search(r'(?:最終評級|評級|等級)[\s：:]*([ABCD])', report, re.IGNORECASE)
        if match:
            grade = match.group(1).upper()
            if grade in ['A', 'B']:
                print(f"✅ 檢查通過 (判定評級: {grade})")
                return True
            else:
                print(f"❌ 檢查未通過 (判定評級: {grade})")
                return False
        else:
            last_part = report[-200:].upper()
            if '評級：A' in last_part or '評級A' in last_part or '等級A' in last_part:
                print("✅ 檢查通過 (預估評級: A)")
                return True
            elif '評級：B' in last_part or '評級B' in last_part or '等級B' in last_part:
                print("✅ 檢查通過 (預估評級: B)")
                return True
            elif '評級：C' in last_part or '評級C' in last_part or '等級C' in last_part:
                print("❌ 檢查未通過 (預估評級: C)")
                return False
            elif '評級：D' in last_part or '評級D' in last_part or '等級D' in last_part:
                print("❌ 檢查未通過 (預估評級: D)")
                return False
            
            print("⚠️ 無法明確解析評級，放棄新增。")
            return False

    except Exception as e:
        print(f"⚠️ 檢查過程發生錯誤: {e}")
        return False


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
                for i, item in enumerate(pairs, 1):
                    if isinstance(item, list):
                        print(f"    [{i:02d}] 多輪對話 ({len(item)} 輪):")
                        for t, (u, a) in enumerate(item, 1):
                            print(f"         T{t} user: {u}")
                            print(f"         T{t} asst: {a}")
                    else:
                        u, a = item
                        print(f"    [{i:02d}] user: {u}")
                        print(f"         Gemini 回答: {a}")
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
        for item in pairs:
            first_user = _get_first_user_msg(item)
            if first_user in exact_set:
                total_rejected += 1
                continue
            norm_hash = hash_key(normalize_for_dedupe(first_user))
            if norm_hash in normalized_set:
                total_rejected += 1
                continue

            exact_set.add(first_user)
            normalized_set.add(norm_hash)
            accepted.append(item)
            batch_accepted += 1
            if len(accepted) >= TARGET_COUNT:
                break

        print(
            f"  ✓ call #{total_calls}: API 回傳 {len(pairs)},"
            f" 接受 {batch_accepted}, 累計 {len(accepted)}/{TARGET_COUNT}"
        )
        time.sleep(SLEEP_BETWEEN_CALLS)

    # 寫入 master 與 latest
    if accepted:
        # 執行品質檢查
        if not evaluate_data_quality(client, accepted):
            print("🛑 因資料品質未達 A 或 B，已放棄此次新增，程式中斷。")
            sys.exit(1)
            
        append_to_master(accepted)
        write_to_latest(accepted)

    print("─" * 60)
    print(f"🎯 完成: 本次新增 {len(accepted)} 筆到 {MASTER_FILE}")
    print(f"   同步覆蓋寫入本次結果至: {LATEST_FILE}")
    print(f"   API 呼叫次數: {total_calls}")
    print(f"   API 回傳總筆數: {total_from_api}")
    print(f"   被去重拒絕筆數: {total_rejected}")
    if len(accepted) < TARGET_COUNT:
        print(f"⚠️  未達目標 {TARGET_COUNT} 筆,可能是 API 失敗次數過多或去重過嚴")


if __name__ == "__main__":
    main()
