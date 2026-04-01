import json
from collections import defaultdict

# ==== 文件路径 ====
review_path = r"D:\workspace\Review_Music\data\Digital_Music_5\Digital_Music_5_sentiment_train.tsv"
aspect_path = r"D:\workspace\dataset\aspect\dm5\11.train_pred_gemma3_filtered.txt"

# ==== 1. 读取 aspect 预测结果 ====
print("Loading aspect predictions...")
pred_lines = []
with open(aspect_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == "none" or not line:
            pred_lines.append(None)
        else:
            try:
                pred_lines.append(json.loads(line))
            except:
                pred_lines.append(None)
print(f"Loaded {len(pred_lines)} lines of aspect predictions")

# ==== 2. 初始化结构 ====
# item -> aspect -> {positive: count, neutral: count, negative: count}
item_aspect_sent = defaultdict(lambda: defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0}))
# item -> aspect -> score (pos - neg)
item_aspect_score = defaultdict(dict)

# ==== 3. 逐行读取 review 文件 ====
print("Processing reviews...")
with open(review_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= len(pred_lines):
            break
        pred = pred_lines[idx]
        if not pred:
            continue

        try:
            record = json.loads(line)
        except:
            continue

        item = record.get("item_id")
        if item is None:
            continue

        # 遍历每个aspect预测
        for key, count in pred.items():
            if not (key.startswith("(") and "," in key):
                continue

            aspect = key.split(",")[0].strip(" (")
            sentiment = key.split(",")[1].strip(" )")

            if sentiment not in ["positive", "neutral", "negative"]:
                continue

            item_aspect_sent[item][aspect][sentiment] += count

# ==== 4. 计算情感得分 ====
for item, aspects in item_aspect_sent.items():
    for aspect, senti_count in aspects.items():
        score = senti_count["positive"] - senti_count["negative"]
        item_aspect_score[item][aspect] = score

# ==== 5. 统计信息 ====
all_items = len(item_aspect_sent)
all_aspects = len({a for aspects in item_aspect_sent.values() for a in aspects})
print(f"共统计到 {all_items} 个 item")
print(f"共出现 {all_aspects} 种 aspect")

# ==== 6. 保存结果 ====
# save_sent_path = r"D:\workspace\dataset\aspect\dm5\12.item_aspect_sentiment.json"
save_score_path = r"D:\workspace\dataset\aspect\dm5\12.item_aspect_total_sent.json"

# with open(save_sent_path, "w", encoding="utf-8") as f:
#     json.dump(item_aspect_sent, f, ensure_ascii=False, indent=2)

with open(save_score_path, "w", encoding="utf-8") as f:
    json.dump(item_aspect_score, f, ensure_ascii=False, indent=2)

# print(f"Saved sentiment dict → {save_sent_path}")
print(f"Saved score dict → {save_score_path}")
