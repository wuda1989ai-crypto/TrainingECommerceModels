#
#  generate_gemini_data.py
#  TrainingECommerceModels
#
#  呼叫 Gemini API 生成電商導購對話,append 到 data/master_conversations.jsonl,
#  並用純 Python 端的統計與評級檢查資料品質,僅 A/B 級才寫入。
#
#  去重策略 (兩階段組合):
#    1. 精確字串: user 訊息與 master 既有完全相同 → 丟棄
#    2. 正規化 hash: NFKC + 去空白 + 去常見標點 + 小寫後做 SHA1 → 丟棄
#
#  環境變數:
#    GEMINI_API_KEY  — 必填
#

import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
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
MODEL_NAME = GeminiModel.GEMINI_31_FLASH_LITE  # ← 切換模型只需改這一行

TARGET_COUNT = 50          # 要 AI 新增的筆數
BATCH_SIZE = 25            # 每次 API 呼叫要幾筆 (控制輸出 token)
MAX_CALLS = 8               # 安全上限,避免去重後補不滿無限呼叫
RETRY_PER_CALL = 2           # 單一 batch 失敗重試次數
SLEEP_BETWEEN_CALLS = 1.5    # 每次呼叫間隔 (秒),友善 rate limit
MIN_USER_LEN = 3             # 過濾過短 user 訊息的最小字數 (短於此直接丟棄)

# ---- Gemini 系統提示 ---------------------------------------------------------
# 為了讓訓練資料貼近「真實電商流量」的樣態,此 prompt 強制 Gemini 在單一 batch
# 內產出多種風格的 user query (關鍵字式、情境式、比較式、規格式、長描述、售後),
# 並嚴格要求 batch 內 user 長度與句式必須有明顯差異,降低語意重複率。
GEMINI_SYSTEM_PROMPT = """你是一位資料集生成助手,專門為繁體中文電商導購模型產生訓練對話。

請生成 {n} 筆繁體中文的電商導購對話,每筆對話包含 "user" 與 "assistant" 兩個欄位。

# 【重要】user query 風格分布 — 本 batch {n} 筆「必須」按下方的**精確筆數**分配,嚴禁某一風格超量或缺席。
# 產出前請先在心裡點算:總數加起來必須等於 {n} 筆。
#
# ⚠️ **這 6 個風格是互斥的!** 每一筆 user 只能屬於**其中一個**,不可重複計算。
# 判定優先順序: 風格 5 (長度 > 60 字) → 風格 6 (售後關鍵字) → 風格 3 (A vs B 比較) →
#              風格 4 (型號 + 規格數字) → 風格 2 (純關鍵字無標點) → 風格 1 (情境式)

1. **情境式完整句子 — 本 batch 恰好 {style_scenario} 筆** — 長度 20~40 字,帶使用者身分或情境,**以完整句子表達**。
   ✅ 是:「我是租屋族,房間小,想要一台不佔空間的吸塵器」
         「下個月要帶爸媽去日本,想買雙好走的運動鞋」
         「家裡剛出生寶寶,想找一款溫和的洗髮精」
   ❌ **不是** (這些屬於別的風格,不要算到情境式):
     - 「Dyson V12 跟 V15 差在哪?」 → 這是**風格 3 比較**
     - 「iPhone 16 Pro Max 256GB 哪個顏色好看?」 → 這是**風格 4 規格**
     - 「無線耳機推薦」「送禮 長輩」 → 這是**風格 2 短關鍵字**
     - 超過 60 字的多條件需求 → 這是**風格 5 長描述**

2. **短關鍵字搜尋 — 本 batch 恰好 {style_short_kw} 筆** — 長度 3~10 字,**完全沒有語氣詞、動詞、標點**,純粹像在 Google 或電商 App 搜尋框打字。
   ✅ 是:「高cp值 nvidia 顯卡」「無線耳機推薦」「便宜吸塵器」
         「送禮 長輩 實用」「iphone 15 pro 256 黑」「dyson v12 保養」
   ❌ **不是**:
     - 「想買無線耳機」 → 有「想買」動詞,屬**風格 1 情境式**
     - 「耳機推薦嗎?」 → 有標點,屬**風格 1 情境式**
   ⚠️ **絕對不可超過 {style_short_kw} 筆**,也不可少於 {style_short_kw} 筆。

3. **品牌/型號比較 — 本 batch 恰好 {style_compare} 筆** — 必須是 **A vs B** 或「XXX 值不值得」的明確比較句。
   ✅ 是:「Dyson V12 跟 V15 差在哪?」
         「Sony WF-1000XM5 跟 Bose QC Ultra 哪個抗噪比較強?」
         「MacBook Air M3 值得買嗎還是等 M4?」
   ❌ **不是**:
     - 「想找一台好用的吸塵器」 → 沒有特定 A/B,屬**風格 1 情境式**
     - 「iPhone 16 Pro 值得買嗎?」(只問一個) → 屬**風格 4 規格**

4. **具體規格/型號詢問 — 本 batch 恰好 {style_spec} 筆** — 明確帶**型號名 + 具體規格** (容量、顏色、版本號、尺寸)。
   ✅ 是:「iPhone 16 Pro Max 256G 黑鈦跟白鈦哪個比較耐看?」
         「PS5 Slim 光碟版現在還買得到嗎?」
         「Switch 2 首發版跟普通版差多少?」
   ❌ **不是**:
     - 「想買 iPhone」(沒規格) → 屬**風格 1 情境式**
     - 「iphone 15 pro 256」(純關鍵字無標點) → 屬**風格 2 短關鍵字**

5. **超長多條件需求 — 本 batch 恰好 {style_long} 筆** — 長度**至少 60 字**,一次列出 3 個以上的限制條件 (預算、坪數、家庭成員、功能需求等)。
   ✅ 是:「我家客廳 15 坪,有兩個小孩會亂跑,想買吸塵器但預算不能超過五千,希望吸力強還要能吸床墊塵蟎,品牌保固也要久一點,請幫我推薦」
   ❌ **不是**:
     - 40 字以下的情境句 → 屬**風格 1 情境式**

6. **售後/使用問題 — 本 batch 恰好 {style_aftersale} 筆** — 已購後的疑問、保固、維修、退換、使用困擾。
   ✅ 是:「藍牙耳機只有一邊有聲音,保固怎麼處理?」
         「昨天下單今天就降價,可以退差價嗎?」
         「Dyson 濾網多久要換一次?」
   ❌ **不是**:
     - 「買前想問...」 → 屬**風格 1 情境式** 或其他購前風格

# 【風格自我檢查】產出前請先在心裡點算:
# 情境式 {style_scenario} + 短關鍵字 {style_short_kw} + 比較 {style_compare} + 規格 {style_spec} + 長描述 {style_long} + 售後 {style_aftersale} = {n} 筆
# 若數量不符、或發現某一類寫太多,請**重新分配**而不是硬湊。

# 【🚨 user 訊息字數硬性配額 — 這是評分工具的真實判定方式】
# 評分工具**只看字數**來分桶 user 訊息,風格名稱對它而言不存在:
#   - 桶 A 「短」:user 字數 **< 10 字** (純關鍵字、無標點)
#   - 桶 B 「中」:user 字數 **10 ~ 40 字** (= 評分工具所說的「情境式」)
#   - 桶 C 「長」:user 字數 **> 40 字**
# 只要桶 B 超過 40%,評分就會 WARN,**不論你心裡覺得它是什麼風格**。
#
# 本 batch {n} 筆的字數配額(請逐筆計算字數,**字數而非風格**才是判定基準):
#   - 桶 A (< 10 字):**至少 {bucket_short} 筆** (約 35%)
#   - 桶 B (10~40 字):**最多 {bucket_mid} 筆** (約 35%)
#   - 桶 C (> 40 字):**至少 {bucket_long} 筆** (約 30%)
#
# ⚠️ 寫每一筆 user 之前先決定要哪個桶,然後**真的數字數**:
#   - 桶 A 範例(全部 < 10 字):「滑鼠 無線」「冷氣 變頻 1 噸」「跑鞋 馬拉松」「Switch 配件」「冰箱 雙門 350L」
#   - 桶 B 範例(10~40 字):「想找一台適合外送員的小折」「冬天露營睡袋推薦保暖款」
#   - 桶 C 範例(> 40 字):「我兒子今年要上大學讀資訊系,預算 4 萬左右,想幫他挑一台筆電,主要會跑 IDE 跟 Docker,偶爾打電動」
#
# 寫到第 {bucket_mid} 筆桶 B 之後,**剩下的 user 一律改寫成桶 A 或桶 C**,絕不可再寫 10~40 字的句子。
# 例:「家裡浴室在漏水想找廠商」(15 字,屬 B) → 改成「浴室抓漏」(4 字,屬 A) 或加長到「桃園中壢透天浴室磁磚下方在漏水,想找有經驗的抓漏師傅,預算 3 萬內」(35 字...等等仍是 B,要改寫到 > 40 字才算 C)。

# 【多輪對話】約 15% 的對話請設計為 2~3 輪往返,模擬真實的追問場景。
格式為 turns 陣列,每個 turn 包含 user 和 assistant:
  {{"turns": [
    {{"user": "筆電推薦 輕薄", "assistant": "輕薄筆電目前熱門的有 MacBook Air M3 和 ASUS Zenbook 14…請問主要用途是?"}},
    {{"user": "主要寫程式跟跑 Docker", "assistant": "如果要跑 Docker,建議 RAM 至少 16GB…"}}
  ]}}
多輪對話的 user 追問應該要自然地延續上一輪的話題,例如補充需求、追問細節、比較選項等。

# 商品類別 — 本次 batch 「只能」從以下類別中挑選商品,不可使用清單外的類別:

{categories}

**類別規則**:
- 本 batch 的 {n} 筆對話必須**至少涵蓋上方 3 個不同主分類**,不可全部集中在同一類。
- 每筆對話請對應其中一個子項目,並可自然延伸到該分類下的具體商品或型號 (例如「吸塵器」→ Dyson V15、小米無線吸塵器)。
- **嚴格禁止**使用上方清單外的商品類別。

# assistant 回覆規範:
1. 以親切、專業的電商導購助手口吻回答。
2. 給出 2~4 個具體商品建議或選購重點,包含簡短理由 (價格帶、適用情境、材質、品牌特色)。
3. **結尾類型 — 雙向硬性配額**(**過低或過高都會 FAIL**):
   - 本 batch {n} 筆中:
     * **問句結尾**目標 **{end_with_question} 筆 (約 60%)**,**不可少於** {end_with_question} 的 80%,**不可多於** {end_with_question} 的 130%
     * **陳述句結尾**目標 **{end_with_statement} 筆 (約 40%)**,**不可少於** {end_with_statement} 的 60%
   - **問句結尾**用法 — 用戶訊息模糊 (短關鍵字、情境不清、需求未定) → 用問句追問:
     * 「請問您比較重視續航還是效能呢?」
     * 「平常主要在家用還是外出?」
     * 「您預算大概多少呢?」
   - **陳述句結尾**用法 — 用戶需求已明確 (帶齊預算/型號/情境) → 直接推薦並用陳述句:
     * 行動建議:「建議您先去門市試坐看看」「可以從 A、B 兩款優先比較」
     * 總結重點:「整體來說 X 款最適合您的需求」「這個價位選 X 是 CP 值最高的」
     * 肯定收尾:「這個選擇很適合您的使用情境」
   - ⚠️ **歷史教訓**(雙邊都犯過,小心):
     * 太多問句 (> 70%) → 訓練後模型「只會問不會答」
     * 太少問句 (< 30%) → 訓練後模型「不會主動追問需求」,失去導購互動性
     * **這次目標就是 60%**,既不要極端高也不要極端低
   - 寫完前先點算:目前問句 X 筆,陳述 Y 筆。X 應接近 {end_with_question},Y 應接近 {end_with_statement}。
4. 回答長度 80~200 字,避免流水帳。
5. 針對風格 2 (短關鍵字) 與 6 (售後) 的 user,assistant 也要合理應對 — 短關鍵字就猜測用戶意圖並給建議,售後就給處理流程並問細節,不要每一筆都假裝用戶給了完整情境。
6. **開頭用語禁令** (非常重要,請逐筆檢查):
   - 本 batch {n} 筆中,以「您好」開頭的**不得超過 {max_you_hao} 筆**。
   - **嚴禁**連續出現以下套版開頭 (整個 batch 加起來也請盡量避免):
     「您好！」「您好,」「嗨!」「好的,」「沒問題,」「當然可以」「很高興為您」
   - 請**優先**使用以下開頭方式,並混合輪替:
     * 直接切入主題 (例:「這兩款各有優勢…」「針對您的需求…」)
     * 肯定用戶的選擇 (例:「很棒的選擇！」「這個預算可以挑到不錯的…」)
     * 情境同理 (例:「租屋族想省空間對吧?…」「賞楓季快到了…」)
     * 直接給答案 (例:「最熱門的三款是…」「先從 A、B、C 三個方向看…」)
7. **只推薦真實存在且知名度高的品牌與型號**。如果不確定某型號是否存在,改用品牌名 + 系列名 (例:「Panasonic 的 NA 系列」),不要自己編造型號。

# 【語言純度】(非常重要,違反者整筆會被 Python 後處理自動捨棄)
- 所有 user 與 assistant 的內容**只允許**使用:繁體中文 + 英文 (英文僅限品牌名、型號、規格單位如 GB/cc/inch)。
- **嚴禁**出現以下任何文字系統:
  * 簡體中文、日文假名 (ひらがな/カタカナ)、韓文 (한글)
  * 阿拉伯文、希伯來文、泰文、俄文、越南文聲調字母
  * **印度語系**:Hindi (देवनागरी)、Gujarati (ગુજરાતી)、Tamil (தமிழ்)、Bengali、Punjabi 等
- **嚴禁**混入任何程式碼片段 (如 `getApplication`、`function()`、HTML tag `<div>`)、JSON 碎片 (如 `"}},{{"`、`!}}{{`)、占位符 (如 `{{placeholder}}`、`[TODO]`)。
- 如需表達「公共科目」「一般類別」等中性詞,**一律使用繁中**,不可用任何外語對譯。
- 若品牌有台灣慣用譯名請優先使用台灣譯名 (例:Nestle→雀巢、L'Oreal→萊雅);iPhone、Dyson、Nike 等英文品牌則保留原文不譯。
- 禁止大陸用語:「視頻」改「影片」、「屏幕」改「螢幕」、「土豆」改「馬鈴薯」、「優盤」改「隨身碟」、「短視頻」改「短影片」。

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


# ---- 商品類別池 (供動態抽樣) -------------------------------------------------
# 每次呼叫 Gemini 時,從此池隨機抽幾個主分類與子項目組成 prompt 區塊。
# 目的:
#   1. 壓縮 prompt token (子項目隨機抽 4×8 而非全塞 12×N)
#   2. 強迫跨 batch 的類別分布平均,避免 Gemini 反覆挑 3C/家電的舒適圈
#   3. 同 batch 限定在少數類別內,主題更聚焦、更容易拉開差異
# 新增/修改商品類別時直接改這個 dict,不需要動 prompt 字串。
# 注意:此 dict 的主類別 key 應與 CATEGORY_KEYWORDS 的 key 對齊,否則統計分桶會錯位。
CATEGORY_POOL = {
    "3C/手機/電腦": [
        "穿戴裝置", "藍牙耳機", "藍牙喇叭", "行動電源", "延長線",
        "滑鼠", "機械鍵盤", "網通分享器", "監視器", "智能居家",
        "iPhone", "安卓手機", "iPad", "平板電腦", "手機殼",
        "微單眼", "單眼鏡頭", "空拍機", "記憶卡", "防潮箱",
        "筆記型電腦", "商用筆電", "DIY組裝電腦", "LCD螢幕", "Mac",
        "電競筆電", "電競電腦", "電競週邊", "遊戲掌機",
        "CPU", "主機板", "SSD", "記憶體", "顯示卡", "NAS", "不斷電系統",
    ],
    "家電": [
        "冰箱", "冷凍櫃", "洗衣機", "滾筒洗衣機", "乾衣機", "冷氣", "移動式空調", "免治馬桶",
        "清淨機", "除濕機", "電暖器", "循環扇", "捕蚊家電",
        "電鍋", "快煮壺", "咖啡機", "飲水設備", "洗碗機", "微波爐", "烤箱", "淨水器", "排油煙機", "熱水器", "瓦斯爐", "氣炸鍋",
        "吸塵器", "掃地機器人", "除蟎機", "蒸氣熨斗",
        "吹風機", "直髮夾", "美容儀", "電動牙刷", "刮鬍刀",
        "液晶電視", "OLED電視", "投影機", "家庭劇院", "PlayStation",
    ],
    "美妝/保養/香氛": [
        "面膜", "化妝水", "精華液", "乳液", "乳霜", "眼霜", "防曬乳",
        "粉底液", "氣墊粉餅", "口紅", "眼影盤", "眉筆", "睫毛膏", "腮紅",
        "香水", "擴香", "精油", "香氛蠟燭",
        "身體乳", "沐浴乳", "護手霜", "髮膜", "洗面乳", "卸妝油",
        "專櫃品牌", "開架品牌", "醫美品牌",
    ],
    "保健/食品": [
        "綜合維他命", "葉黃素", "魚油", "益生菌", "膠原蛋白", "鈣片", "B群",
        "蛋白粉", "代餐", "酵素",
        "口罩", "OK繃", "護具", "熱敷墊", "血壓計",
        "橄欖油", "醬油", "有機食品", "泡麵", "米", "麵條",
        "洋芋片", "餅乾", "巧克力", "氣泡水", "果汁", "礦泉水",
        "濾掛咖啡", "茶包", "麥片", "滴雞精",
        "水產", "牛肉", "豬肉", "蔬菜", "水果", "水餃", "冷凍調理包",
    ],
    "流行時尚/男女裝/鞋包": [
        "T恤", "襯衫", "洋裝", "長褲", "裙子", "風衣", "羽絨外套", "睡衣",
        "女內衣", "無鋼圈內衣", "泳裝",
        "女後背包", "女斜背包", "手提包", "女皮夾",
        "女運動鞋", "樂福鞋", "涼鞋", "高跟鞋", "短靴",
        "男POLO衫", "男背心", "男西裝", "男短褲", "男內褲",
        "男斜背包", "男皮夾",
        "男休閒鞋", "男運動鞋", "紳士鞋", "男拖鞋",
    ],
    "精品/珠寶/手錶": [
        "精品包", "精品皮夾", "精品飾品", "太陽眼鏡", "光學眼鏡",
        "機械錶", "石英錶", "智慧錶", "瑞士錶",
        "黃金項鍊", "黃金手鍊", "鑽戒", "珍珠項鍊", "銀飾", "玉石手鐲",
    ],
    "母嬰/玩具/清潔": [
        "嬰兒紙尿褲", "濕紙巾", "嬰兒推車", "安全座椅", "背巾",
        "奶瓶", "副食品餐具", "嬰兒浴盆", "嬰兒床", "嬰兒寢具",
        "積木", "模型公仔", "桌遊", "拼圖", "扮家家酒玩具",
        "兒童地墊", "兒童書桌", "兒童書包",
        "童裝", "童鞋",
        "洗髮精", "沐浴乳", "牙膏", "牙刷", "洗手乳", "手工皂", "男仕洗面乳",
    ],
    "圖書/文具/影音": [
        "商業理財書", "心理勵志書", "小說", "漫畫", "輕小說", "語言學習書", "童書", "繪本", "考用書",
        "電子書", "電子書閱讀器", "有聲書",
        "原子筆", "鋼筆", "筆記本", "便利貼", "美術繪畫", "辦公用品", "樂器",
        "雜誌", "藝術品",
    ],
    "家具/寢具/居家布置": [
        "沙發", "布沙發", "床架", "雙人床架", "茶几", "餐桌", "化妝台", "電腦椅", "辦公椅",
        "收納櫃", "收納箱", "層架", "壓縮袋",
        "枕頭", "羽絨被", "涼被", "床墊", "乳膠床墊", "床包組",
        "地毯", "窗簾", "抱枕", "毛巾", "掛鐘",
        "開運風水", "鹽燈", "佛珠",
    ],
    "運動/健身/按摩": [
        "運動T恤", "運動短褲", "瑜珈褲", "慢跑鞋", "籃球鞋",
        "啞鈴", "槓鈴", "壺鈴", "瑜珈墊", "運動護膝", "跳繩",
        "棒球手套", "網球拍", "羽球拍", "高爾夫球桿", "登山杖",
        "健身車", "跑步機",
        "按摩椅", "按摩槍", "泡腳機", "眼部按摩器", "筋膜槍",
    ],
    "居家/餐廚/寵物": [
        "衛生紙", "衛生棉", "成人紙尿褲",
        "洗衣精", "洗衣粉", "洗碗精", "地板清潔劑", "除塵蟎", "除濕劑",
        "不沾鍋", "炒鍋", "保溫瓶", "保鮮盒", "保鮮膜", "刀具", "砧板", "馬克杯", "茶壺",
        "狗飼料", "貓飼料", "貓砂", "寵物零食", "寵物玩具", "寵物美容", "水族箱",
        "LED燈泡", "吸頂燈", "電鑽", "園藝工具", "油漆",
    ],
    "戶外/汽機車/旅遊票券": [
        "帳篷", "睡袋", "充氣床", "摺疊桌椅", "卡式爐", "行李箱", "登山背包", "雨衣",
        "電動自行車", "電動滑板車", "機車安全帽",
        "行車記錄器", "汽車芳香", "輪胎", "車用吸塵器", "車內收納",
        "國內旅遊", "訂房券", "溫泉券",
        "下午茶券", "吃到飽券", "飯店住宿券", "SPA按摩券",
    ],
}


def build_categories_block(k_main: int = 4, k_sub: int = 8) -> str:
    """
    隨機從 CATEGORY_POOL 抽 k_main 個主分類,每個主分類再抽 k_sub 個子項目,
    組成 prompt 可用的 markdown bullet 區塊。
    """
    picked_mains = random.sample(list(CATEGORY_POOL.keys()), k=k_main)
    lines = []
    for main in picked_mains:
        subs = CATEGORY_POOL[main]
        sample_n = min(k_sub, len(subs))
        picked_subs = random.sample(subs, k=sample_n)
        lines.append(f"* **{main}**:{'、'.join(picked_subs)}")
    return "\n".join(lines)


# ---- 正規化 (用於 hash 去重) -------------------------------------------------
_PUNCT_PATTERN = re.compile(
    r"[\s、,。.!?！？…~～\-—_/\\\"'`“”‘’「」『』()（）\[\]【】{}《》<>:;：;·]+"
)


def normalize_for_dedupe(text: str) -> str:
    """
    去除空白、全形轉半形、去常見標點、小寫化,
    用於去重比對的 hash key,讓「無線耳機,推薦」與「無線耳機推薦」視為同一筆。
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


# ---- 異常字元偵測 ------------------------------------------------------------
# 攔截 Gemini 偶爾吐出的阿拉伯文、印度語系、日韓假名、程式碼殘渣等雜訊。
# 採白名單策略:列出**允許**的 Unicode 區段 (繁中 + 英數 + 常見標點 + emoji),
# 不在白名單的字元一律視為異常,整筆資料丟棄。
_CONTENT_WHITELIST = re.compile(
    r"^["
    r"\u4e00-\u9fff"          # CJK 基本漢字 (繁中 + 簡中 — 簡中由 prompt 端把關)
    r"\u3400-\u4dbf"          # CJK 擴展 A
    r"\uf900-\ufaff"          # CJK 相容漢字
    r"a-zA-Z0-9"              # 英數
    r"\s"                     # 空白 (含 \n \t)
    r".,;:!?\-_+=*/<>#@&%$^~`|\\()\[\]{}\"'"  # 半形標點
    r"\u3000-\u303f"          # CJK 標點符號區
    r"\uff00-\uffef"          # 全形英數與標點
    r"\u2010-\u2027\u2030-\u205e"  # 一般標點 (破折號、引號、省略號等)
    r"\u00a0-\u00ff"          # Latin-1 補充 (含 °、±、©、§ 等)
    r"\u2100-\u214f"          # 字母類符號 (℃、№、™ 等)
    r"\u2190-\u21ff"          # 箭頭
    r"\u2200-\u22ff"          # 數學運算符
    r"\u2460-\u24ff"          # 圓圈數字
    r"\u25a0-\u25ff"          # 幾何圖形
    r"\u2600-\u27bf"          # 雜項符號與裝飾 (含 ☆ ★ ♥ 等)
    r"\U0001f300-\U0001faff"  # Emoji (補完整段涵蓋率)
    r"]+$"
)


def find_exotic_chars(text: str) -> list[str]:
    """
    掃過 text,回傳所有不在白名單的字元 (去重後最多 5 個);
    回傳空 list 代表內容乾淨。截斷在 5 個是讓 log 訊息可讀。
    """
    if not text:
        return []
    if _CONTENT_WHITELIST.match(text):
        return []
    bad = []
    seen = set()
    for ch in text:
        if not _CONTENT_WHITELIST.match(ch) and ch not in seen:
            seen.add(ch)
            bad.append(ch)
            if len(bad) >= 5:
                break
    return bad


# ---- Master 檔讀寫 ------------------------------------------------------------
def load_master_dedupe_sets():
    """
    讀取既有 master_conversations.jsonl 建立兩組去重比對集合:
      exact_set      — 完整 user 字串(用於精確比對)
      normalized_set — 正規化後的 SHA1 hash(用於忽略標點/大小寫的近似比對)
    多輪對話以第一輪的 user 為去重基準。
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
    categories_block = build_categories_block()
    # 「您好」開頭上限:約 8% 且至少 1 筆,避免套版開頭過度集中
    max_you_hao = max(1, round(n * 0.08))
    # 風格分布精確筆數 — 情境式比例壓低,避免擠壓其他風格
    # 比例:情境 20% / 短關鍵字 30% / 比較 18% / 規格 18% / 長描述 9% / 售後 5%
    style_counts = {
        "style_scenario":  round(n * 0.20),
        "style_short_kw":  round(n * 0.30),
        "style_compare":   round(n * 0.18),
        "style_spec":      round(n * 0.18),
        "style_long":      round(n * 0.09),
        "style_aftersale": max(1, round(n * 0.05)),
    }
    # 修正尾數差額加到「短關鍵字」(比例最高,容錯空間大)
    diff = n - sum(style_counts.values())
    style_counts["style_short_kw"] += diff
    # 結尾類型配額 — 60% 問句 + 40% 陳述,雙向 bound 防偏移
    end_with_question = round(n * 0.60)
    end_with_statement = n - end_with_question
    # user 字數分桶配額 — 對齊 _compute_metrics 的純字數分桶 (< 10 / 10~40 / > 40)
    bucket_short = round(n * 0.35)   # < 10 字
    bucket_mid   = round(n * 0.35)   # 10~40 字
    bucket_long  = n - bucket_short - bucket_mid  # > 40 字
    prompt = GEMINI_SYSTEM_PROMPT.format(
        n=n,
        categories=categories_block,
        max_you_hao=max_you_hao,
        end_with_question=end_with_question,
        end_with_statement=end_with_statement,
        bucket_short=bucket_short,
        bucket_mid=bucket_mid,
        bucket_long=bucket_long,
        **style_counts,
    )
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

    # Batch 內後處理 — 三道過濾,任一違規整筆丟棄:
    #   (a) 過濾過短 user (< MIN_USER_LEN 字元)
    #   (b) batch 內正規化去重 (防 Gemini 同一次呼叫吐出近義改寫)
    #   (c) 異常字元過濾 (阿拉伯/日韓/印度語系/程式碼殘渣)
    # 拒絕統計會印出,方便觀察 prompt 是否需要調整。
    filtered = []
    seen_norms = set()
    reject_stats = {"too_short": 0, "batch_dup": 0, "exotic": 0}

    def _iter_texts(it):
        """產出 item 所有 user + assistant 字串,供內容檢查使用。"""
        if isinstance(it, list):
            for u, a in it:
                yield u
                yield a
        else:
            yield it[0]
            yield it[1]

    for item in pairs:
        first_user = item[0][0] if isinstance(item, list) else item[0]

        # (a) 過短 user
        if len(first_user) < MIN_USER_LEN:
            print(f"    ⚠️ 丟棄過短 user ({len(first_user)}字): {first_user!r}")
            reject_stats["too_short"] += 1
            continue

        # (b) batch 內重複
        norm = hash_key(normalize_for_dedupe(first_user))
        if norm in seen_norms:
            print(f"    ⚠️ 丟棄 batch 內重複 user: {first_user!r}")
            reject_stats["batch_dup"] += 1
            continue

        # (c) 異常字元 — 掃描 item 內所有文本
        exotic_hit = None
        for text in _iter_texts(item):
            bad = find_exotic_chars(text)
            if bad:
                exotic_hit = (text[:40], bad)
                break
        if exotic_hit:
            snippet, bad_chars = exotic_hit
            print(f"    ⚠️ 丟棄含異常字元 {bad_chars!r}: {snippet!r}...")
            reject_stats["exotic"] += 1
            continue

        seen_norms.add(norm)
        filtered.append(item)

    # 印出本 batch 拒絕統計
    total_rejected = sum(reject_stats.values())
    if total_rejected > 0:
        print(
            f"    📊 本 batch 拒絕統計: "
            f"過短={reject_stats['too_short']}, "
            f"batch內重複={reject_stats['batch_dup']}, "
            f"異常字元={reject_stats['exotic']} "
            f"(總拒絕 {total_rejected}/{len(pairs)})"
        )
    return filtered


# ---- 資料審查 (純 Python 統計與評級) -----------------------------------------
# 設計原則:所有判定都用 Python 機械化執行,完全不依賴 LLM。
# 規則型檢查 (字數分桶、結尾問句偵測、分類關鍵字、開頭用語) 用 LLM 來做容易
# 出現:閾值附近自行加軟性判定、先寫標籤再算數值、分類誤分等問題,而 Python
# 直接算數字 100% 準確且零成本。

# 商品分類關鍵字 — 對齊 CATEGORY_POOL 12 主類別,讓統計能用 user 訊息精準歸類。
# 沒命中的歸「其他」;若「其他」比例超過 20% 代表關鍵字漏掉了,請補進對應類別。
# 順序原則:愈具體的關鍵字放前面,避免通用詞先攔截掉特定品牌/型號。
CATEGORY_KEYWORDS = [
    ("3C/手機/電腦", [
        "手機", "iPhone", "安卓", "平板", "iPad", "筆電", "筆記型電腦", "Mac", "MacBook",
        "桌機", "電腦", "螢幕", "LCD", "顯示器", "鍵盤", "滑鼠", "耳機", "藍牙耳機",
        "喇叭", "音響", "藍牙喇叭", "行動電源", "充電器", "充電線", "延長線",
        "穿戴", "智慧手錶", "手環", "監視器", "網通", "分享器", "路由器", "wifi", "Wi-Fi",
        "相機", "微單", "單眼", "鏡頭", "空拍", "記憶卡", "硬碟", "SSD", "NAS",
        "SD卡", "USB", "顯示卡", "顯卡", "CPU", "主機板", "記憶體", "RAM",
        "電競", "遊戲機", "Switch", "PS5", "PS4", "Xbox", "掌機", "GPU",
        # 常見耳機/音響品牌型號(query 沒寫「耳機」時也能命中)
        "AirPods", "Sony WF", "Sony WH", "Sony LinkBuds", "Bose QC", "Bose QuietComfort",
        "Sennheiser", "森海塞爾", "Beats", "鐵三角", "Audio-Technica",
    ]),
    ("家電", [
        "冰箱", "冷凍", "洗衣機", "烘衣", "乾衣", "冷氣", "空調", "暖氣", "電暖",
        "清淨機", "空氣清淨", "除濕", "加濕", "電風扇", "循環扇", "捕蚊",
        "電鍋", "壓力鍋", "氣炸鍋", "烤箱", "微波爐", "咖啡機", "快煮壺", "果汁機",
        "豆漿機", "電磁爐", "瓦斯爐", "排油煙", "抽油煙", "洗碗機", "淨水器",
        "熱水器", "免治馬桶", "吸塵器", "掃地機", "掃地機器人", "除蟎",
        "蒸氣熨斗", "熨斗", "吹風機", "直髮", "電捲", "電動牙刷", "刮鬍刀",
        # 常見家電品牌
        "Dyson", "戴森", "Panasonic", "國際牌", "象印", "膳魔師", "禾聯", "聲寶",
        "iRobot", "Roborock", "石頭科技", "LG 冷氣", "LG 洗衣", "Hitachi", "日立",
    ]),
    ("美妝/保養/香氛", [
        "保養", "面膜", "精華液", "化妝水", "乳液", "乳霜", "防曬", "卸妝",
        "彩妝", "口紅", "唇膏", "粉底", "腮紅", "眼影", "睫毛膏", "眉筆",
        "香水", "香氛", "古龍水", "髮膜", "護髮", "洗髮精", "潤髮", "沐浴乳",
        "牙膏", "刮鬍膏", "美容儀",
    ]),
    ("保健/食品", [
        "保健", "維他命", "益生菌", "葉黃素", "魚油", "鈣", "膠原蛋白", "蛋白粉",
        "蛋白飲", "代餐", "燕窩", "雞精", "滴雞精",
        "零食", "餅乾", "巧克力", "糖果", "果乾", "堅果",
        "茶葉", "咖啡豆", "茶包", "咖啡粉", "速食",
        "保久乳", "牛奶", "豆漿", "果汁",
    ]),
    ("流行時尚/服飾鞋包", [
        "服飾", "上衣", "T恤", "襯衫", "外套", "夾克", "毛衣", "針織",
        "洋裝", "連身裙", "裙子", "短褲", "長褲", "牛仔褲", "西裝",
        "內衣", "內褲", "睡衣", "襪子", "圍巾",
        "鞋", "球鞋", "皮鞋", "高跟鞋", "涼鞋", "拖鞋", "靴",
        "皮夾", "皮帶", "包包", "後背包", "肩背包", "側背包", "錢包",
    ]),
    ("精品/珠寶/手錶", [
        "手錶", "腕錶", "手鍊", "項鍊", "戒指", "耳環", "鑽戒", "黃金", "珠寶",
        "名牌包", "精品", "奢侈", "Hermes", "LV", "Chanel", "Gucci", "Rolex", "Omega",
    ]),
    ("母嬰/玩具/清潔", [
        "嬰兒", "嬰幼", "新生兒", "寶寶", "嬰兒車", "推車", "嬰兒床", "床邊床",
        "尿布", "奶粉", "奶瓶", "副食品", "學步", "圍兜", "口水巾",
        "兒童", "小孩", "童裝", "背巾", "揹巾", "兒童安全座椅", "汽座",
        "玩具", "積木", "拼圖", "公仔", "扭蛋", "模型",
        "洗衣精", "柔軟精", "漂白", "洗碗精", "清潔劑", "除菌", "酒精",
        "衛生紙", "面紙", "捲筒", "廚房紙巾",
    ]),
    ("圖書/文具/影音", [
        "書", "書籍", "小說", "漫畫", "雜誌", "繪本",
        "文具", "辦公用品", "辦公文具", "筆記本", "原子筆", "鋼筆", "鉛筆", "螢光筆", "便條",
        "膠帶", "資料夾", "白板", "計算機", "事務機",
        "CD", "DVD", "藍光", "黑膠", "唱片",
        "樂器", "吉他", "鋼琴", "電子琴", "鼓",
    ]),
    ("家具/寢具/居家", [
        "沙發", "茶几", "書桌", "辦公椅", "電競椅", "餐桌", "餐椅", "椅子",
        "床架", "床墊", "衣櫃", "鞋櫃", "書櫃", "收納櫃", "收納箱",
        "棉被", "被套", "枕頭", "床包", "床單", "毛毯", "蓋毯", "羽絨被", "羽毛枕", "鵝毛",
        "窗簾", "地毯", "踏墊", "燈具", "檯燈", "立燈", "吊燈",
        "鏡子", "穿衣鏡",
        "掛鐘", "時鐘", "鬧鐘", "壁鐘",
    ]),
    ("運動/健身/按摩", [
        "跑步", "跑鞋", "健身", "啞鈴", "槓鈴", "瑜伽", "瑜珈", "彈力帶",
        "腳踏車", "單車", "自行車", "公路車", "登山車",
        "登山", "爬山", "健行", "球類", "籃球", "足球", "排球", "羽球", "桌球", "網球",
        "高爾夫", "游泳", "蛙鏡", "泳衣", "泳褲",
        "按摩椅", "按摩槍", "按摩枕", "筋膜槍",
    ]),
    ("居家餐廚/寵物", [
        "鍋具", "炒鍋", "湯鍋", "平底鍋", "刀具", "菜刀", "砧板",
        "餐具", "碗", "盤", "杯", "馬克杯", "保溫瓶", "水壺",
        "茶壺", "茶具", "茶杯", "茶碗", "手沖", "濾杯",
        "便當盒", "保鮮盒", "密封罐",
        "寵物", "貓", "狗", "飼料", "貓砂", "貓抓板", "寵物窩", "寵物零食",
        "水族", "魚缸", "水族箱", "水族設備", "魚飼料",
    ]),
    ("戶外/汽機車/旅遊", [
        "露營", "帳篷", "睡袋", "登山包", "露營椅", "露營桌", "野餐",
        "汽車", "車用", "行車記錄", "輪胎", "雨刷", "後座",
        "機車", "安全帽", "騎士",
        "電動滑板", "滑板車", "電動代步", "平衡車", "代步車",
        "行李箱", "登機箱", "旅行袋", "旅遊", "票券", "住宿券",
        "溫泉", "飯店", "民宿", "渡假村", "住宿", "湯屋", "包車", "高鐵票", "機票",
    ]),
]


def _classify_category(text: str) -> str:
    """根據文字找出所屬主類別 (找第一個命中)。沒命中回傳『其他』。"""
    for cat, kws in CATEGORY_KEYWORDS:
        for kw in kws:
            if kw in text:
                return cat
    return "其他"


# 中文問句結尾偵測 — 兩種命中條件:
#   1. 末尾是半形 ? 或全形 ?(\uFF1F) 標點
#   2. 末段出現「嗎/呢/嘛/吧」這類疑問語氣詞 (即使沒打問號也算問句)
_QUESTION_TAIL = re.compile(r'[?\uFF1F]\s*$|[嗎呢嘛吧][。.!\uFF01,\uFF0C]?\s*$')


def _ends_with_question(text: str) -> bool:
    if not text:
        return False
    tail = text.rstrip()[-10:]  # 只看末 10 字,效能與準確度的折衷
    return bool(_QUESTION_TAIL.search(tail))


def _extract_user_assistant_pairs(item):
    """從一筆 pair (單輪 tuple 或多輪 list) 抽出 (user, assistant) 序列。"""
    if isinstance(item, list):
        return [(u, a) for u, a in item]
    return [(item[0], item[1])]


def _compute_metrics(pairs):
    """掃過全部 pairs,計算所有評級需要的統計指標,回傳一個 dict。"""
    n = len(pairs)
    user_lens = []
    asst_lens = []
    asst_starts = Counter()  # assistant 開頭前 4 字的計數
    asst_q_count = 0
    asst_total = 0
    user_short = 0   # < 10 字
    user_mid = 0     # 10~40 字
    user_long = 0    # > 40 字
    user_too_short = 0   # < 3 字 (品質問題)
    asst_too_short = 0   # < 30 字 (品質問題)
    user_seen = set()
    user_seen_norm = set()
    dup_exact = 0
    dup_norm = 0
    cat_counter = Counter()
    multi_count = 0
    single_count = 0

    for item in pairs:
        ua_seq = _extract_user_assistant_pairs(item)
        if len(ua_seq) >= 2:
            multi_count += 1
        else:
            single_count += 1
        # 商品分類用第一輪 user 訊息做關鍵字判定
        first_user = ua_seq[0][0]
        cat_counter[_classify_category(first_user)] += 1

        for u, a in ua_seq:
            ulen = len(u.strip())
            alen = len(a.strip())
            user_lens.append(ulen)
            asst_lens.append(alen)
            if ulen < 3:
                user_too_short += 1
            if alen < 30:
                asst_too_short += 1
            if ulen < 10:
                user_short += 1
            elif ulen <= 40:
                user_mid += 1
            else:
                user_long += 1
            if u in user_seen:
                dup_exact += 1
            else:
                user_seen.add(u)
            norm = normalize_for_dedupe(u)
            if norm in user_seen_norm:
                dup_norm += 1
            else:
                user_seen_norm.add(norm)
            asst_total += 1
            if _ends_with_question(a):
                asst_q_count += 1
            asst_starts[a.strip()[:4]] += 1

    total_user = len(user_lens)
    total_asst = len(asst_lens)

    return {
        "total_records": n,
        "total_user_msgs": total_user,
        "total_asst_msgs": total_asst,
        "single_count": single_count,
        "multi_count": multi_count,
        "user_len": {
            "min": min(user_lens) if user_lens else 0,
            "max": max(user_lens) if user_lens else 0,
            "avg": round(sum(user_lens) / total_user, 1) if total_user else 0.0,
        },
        "asst_len": {
            "min": min(asst_lens) if asst_lens else 0,
            "max": max(asst_lens) if asst_lens else 0,
            "avg": round(sum(asst_lens) / total_asst, 1) if total_asst else 0.0,
        },
        "asst_top_starts": asst_starts.most_common(5),
        "asst_q_count": asst_q_count,
        "asst_q_ratio": asst_q_count / total_asst if total_asst else 0.0,
        "user_buckets": {
            "short": user_short,
            "mid": user_mid,
            "long": user_long,
        },
        "user_bucket_ratios": {
            "short": user_short / total_user if total_user else 0.0,
            "mid": user_mid / total_user if total_user else 0.0,
            "long": user_long / total_user if total_user else 0.0,
        },
        "user_too_short": user_too_short,
        "asst_too_short": asst_too_short,
        "dup_exact": dup_exact,
        "dup_norm": dup_norm,
        "categories": cat_counter,
    }


def _grade_from_metrics(m):
    """
    依機械化規則對 metrics 評級。回傳 (grade, problems, n_warn, n_fail)。
    每條規則會 push 一筆 (status, label, detail) 到 problems。

    評級表:
      A — 全部 PASS
      B — 至少 1 個 WARN,但無 FAIL
      C — 1 個 FAIL
      D — 2 個或以上 FAIL
    """
    problems = []

    def add(status, label, detail):
        problems.append((status, label, detail))

    # 規則 1:user 訊息重複 (任一筆重複即 WARN)
    total_dup = m["dup_exact"] + m["dup_norm"]
    add("PASS" if total_dup == 0 else "WARN",
        "重複 user 訊息",
        f"完全重複 {m['dup_exact']} 筆 + 正規化重複 {m['dup_norm']} 筆")

    # 規則 2:assistant 回覆過短 (< 30 字)
    add("PASS" if m["asst_too_short"] == 0 else "WARN",
        "assistant 過短 (< 30 字)",
        f"{m['asst_too_short']} 筆")

    # 規則 3:user 訊息過短 (< 3 字)
    add("PASS" if m["user_too_short"] == 0 else "WARN",
        "user 過短 (< 3 字)",
        f"{m['user_too_short']} 筆")

    # 規則 4:開頭用語集中度 — 單一 pattern > 40% 即 WARN
    if m["asst_top_starts"]:
        top_pat, top_cnt = m["asst_top_starts"][0]
        top_ratio = top_cnt / m["total_asst_msgs"]
        if top_ratio > 0.40:
            add("WARN", "開頭用語過度集中",
                f"「{top_pat}」佔 {top_ratio*100:.1f}% (> 40%)")
        else:
            add("PASS", "開頭用語",
                f"最高「{top_pat}」佔 {top_ratio*100:.1f}% (≤ 40%)")

    # 規則 5:結尾問句比例 — 雙向門檻
    #   > 85% FAIL (訓練後只會問不會答)
    #   70~85% WARN (偏高)
    #   ≤ 30% WARN (偏低,失去追問互動)
    #   30%~70% PASS
    q_ratio = m["asst_q_ratio"]
    pct = q_ratio * 100
    if q_ratio > 0.85:
        add("FAIL", "結尾問句比例過高",
            f"{pct:.1f}% > 85% (模型訓練後會只問不答)")
    elif q_ratio > 0.70:
        add("WARN", "結尾問句比例偏高", f"{pct:.1f}% (70~85%)")
    elif q_ratio <= 0.30:
        add("WARN", "結尾問句比例偏低",
            f"{pct:.1f}% ≤ 30% (模型訓練後不會主動追問)")
    else:
        add("PASS", "結尾問句比例", f"{pct:.1f}% (30%~70%)")

    # 規則 6:商品主類別集中度 — 排除「其他」後,最高類別 > 30% 即 WARN
    main_cats = [(c, n) for c, n in m["categories"].items() if c != "其他"]
    if main_cats:
        top_cat, top_n = max(main_cats, key=lambda x: x[1])
        top_ratio = top_n / m["total_records"]
        if top_ratio > 0.30:
            add("WARN", "類別分布過於集中",
                f"「{top_cat}」佔 {top_ratio*100:.1f}% (> 30%)")
        else:
            add("PASS", "類別分布",
                f"最高「{top_cat}」佔 {top_ratio*100:.1f}% (≤ 30%)")

    # 規則 7:「其他」類別比例 — > 20% 代表分類關鍵字字典漏掉,需補進去
    other_n = m["categories"].get("其他", 0)
    other_ratio = other_n / m["total_records"]
    if other_ratio > 0.20:
        add("WARN", "「其他」類別比例過高",
            f"{other_ratio*100:.1f}% > 20% (代表分類關鍵字漏掉)")
    else:
        add("PASS", "「其他」類別比例",
            f"{other_ratio*100:.1f}% (≤ 20%)")

    # 規則 8:user 字數分桶分布 — 任一桶超出寬鬆門檻即 WARN
    #   短 (< 10 字) > 60% / 中 (10~40 字) > 70% / 長 (> 40 字) > 60%
    #   中桶門檻較寬,因為 LLM 對中文字數控制不可靠,且現實電商查詢多落此區間
    br = m["user_bucket_ratios"]
    if br["short"] > 0.60:
        add("WARN", "user 短關鍵字 (< 10 字) 過高", f"{br['short']*100:.1f}% > 60%")
    elif br["mid"] > 0.70:
        add("WARN", "user 情境式 (10~40 字) 過高", f"{br['mid']*100:.1f}% > 70%")
    elif br["long"] > 0.60:
        add("WARN", "user 長描述 (> 40 字) 過高", f"{br['long']*100:.1f}% > 60%")
    else:
        add("PASS", "user 風格分布",
            f"短 {br['short']*100:.0f}% / 中 {br['mid']*100:.0f}% / 長 {br['long']*100:.0f}%")

    # 計算 WARN/FAIL 數量並決定評級
    n_fail = sum(1 for s, _, _ in problems if s == "FAIL")
    n_warn = sum(1 for s, _, _ in problems if s == "WARN")
    if n_fail >= 2:
        grade = "D"
    elif n_fail == 1:
        grade = "C"
    elif n_warn > 0:
        grade = "B"
    else:
        grade = "A"
    return grade, problems, n_warn, n_fail


def _format_report(m, grade, problems, n_records):
    """把 metrics + problems 組成 markdown 報告字串。"""
    lines = []
    lines.append("## 訓練資料品質報告 (Python 機械評級)")
    lines.append("")
    lines.append(f"**最終評級:{grade}**")
    lines.append("")
    lines.append(f"檔案:inline ({n_records} 筆)")
    lines.append(f"檢查時間:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("### 內容統計")
    lines.append("| 指標 | 數值 |")
    lines.append("|------|------|")
    lines.append(f"| 總筆數 | {m['total_records']} |")
    lines.append(f"| user 訊息長度 (最短/平均/最長) | "
                 f"{m['user_len']['min']} / {m['user_len']['avg']} / {m['user_len']['max']} 字 |")
    lines.append(f"| assistant 回覆長度 (最短/平均/最長) | "
                 f"{m['asst_len']['min']} / {m['asst_len']['avg']} / {m['asst_len']['max']} 字 |")
    lines.append(f"| 單輪 / 多輪 | {m['single_count']} / {m['multi_count']} |")
    lines.append("")
    lines.append("### 多樣性分析")
    starts_str = ", ".join(f"{p} ({c} 筆)" for p, c in m["asst_top_starts"])
    lines.append(f"- **assistant 開頭 (前 5 名)**: {starts_str}")
    lines.append(f"- **assistant 結尾問句**: {m['asst_q_count']}/{m['total_asst_msgs']} "
                 f"({m['asst_q_ratio']*100:.1f}%)")
    br = m["user_bucket_ratios"]
    ub = m["user_buckets"]
    lines.append(
        f"- **user 風格分布**: 短關鍵字 {ub['short']} ({br['short']*100:.0f}%), "
        f"情境式 {ub['mid']} ({br['mid']*100:.0f}%), "
        f"長描述 {ub['long']} ({br['long']*100:.0f}%)"
    )
    cat_str = ", ".join(
        f"{c} {n} ({n/m['total_records']*100:.0f}%)"
        for c, n in m["categories"].most_common()
    )
    lines.append(f"- **商品類別分布**: {cat_str}")
    lines.append("")
    lines.append("### 潛在問題")
    for status, label, detail in problems:
        lines.append(f"- [{status}] {label}: {detail}")
    lines.append("")
    lines.append("### 總結")
    if grade == "A":
        lines.append("✨ 資料品質良好,所有檢查項目皆 PASS,可直接用於訓練。")
    elif grade == "B":
        lines.append(f"資料格式正確,但有 {sum(1 for s,_,_ in problems if s=='WARN')} 個 WARN,可訓練但建議改善。")
    elif grade == "C":
        lines.append("有 1 個 FAIL,建議修正後再訓練。")
    else:
        lines.append("有多個 FAIL,資料不適合直接訓練。")
    return "\n".join(lines)


def evaluate_data_quality(pairs):
    """
    對 pairs 做品質檢查並印出報告。
    回傳 True 代表通過 (A 或 B 級),呼叫端可以安心寫入 master;
    回傳 False 代表未通過 (C 或 D 級),呼叫端應放棄這批資料。
    """
    print("\n🧐 正在執行 Python 端資料品質檢查...", flush=True)
    metrics = _compute_metrics(pairs)
    grade, problems, n_warn, n_fail = _grade_from_metrics(metrics)
    report = _format_report(metrics, grade, problems, len(pairs))

    print("\n" + "="*20 + " 品質檢查報告 " + "="*20)
    print(report)
    print("="*54 + "\n")

    if grade in ('A', 'B'):
        print(f"✅ 檢查通過 (判定評級: {grade}, WARN={n_warn}, FAIL={n_fail})")
        return True
    else:
        print(f"❌ 檢查未通過 (判定評級: {grade}, WARN={n_warn}, FAIL={n_fail})")
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

    # 品質檢查通過後才寫入 master 與 latest;C/D 級直接放棄,避免污染資料集
    if accepted:
        if not evaluate_data_quality(accepted):
            print("🛑 因資料品質未達 A 或 B,已放棄此次新增,程式中斷。")
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
