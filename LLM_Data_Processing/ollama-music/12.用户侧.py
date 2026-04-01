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

# ==== 2. 初始化统计结构 ====
# user -> aspect -> {pos: count, neutral: count, neg: count}
user_aspect_sent = defaultdict(lambda: defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0}))

# user -> aspect -> total_count
user_aspect_total = defaultdict(lambda: defaultdict(int))

# ==== 3. 逐行读取原始 review 文件 ====
print("Processing reviews...")
with open(review_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= len(pred_lines):
            break  # 保证行数一致
        pred = pred_lines[idx]
        if not pred:
            continue

        try:
            record = json.loads(line)
        except:
            continue

        user = record.get("user_id")
        if user is None:
            continue

        # 遍历每个 aspect-sentiment
        for key, count in pred.items():
            if not (key.startswith("(") and "," in key):
                continue
            aspect = key.split(",")[0].strip(" (")
            sentiment = key.split(",")[1].strip(" )")

            if sentiment not in ["positive", "neutral", "negative"]:
                continue

            user_aspect_sent[user][aspect][sentiment] += count
            user_aspect_total[user][aspect] += count

# ==== 4. 统计所有出现过的 aspect ====
all_aspects = set()
for user, aspects in user_aspect_total.items():
    all_aspects.update(aspects.keys())

# ==== 5. 输出统计结果 ====
print(f"共统计到 {len(user_aspect_sent)} 个用户")
print(f"共出现 {len(all_aspects)} 种 aspect")

# ==== 6. 可选：保存结果 ====
save_user_aspect_sent = r"D:\workspace\dataset\aspect\dm5\12.user_aspect_sentiment.json"
save_user_aspect_total = r"D:\workspace\dataset\aspect\dm5\12.user_aspect_total.json"

with open(save_user_aspect_sent, "w", encoding="utf-8") as f:
    json.dump(user_aspect_sent, f, ensure_ascii=False, indent=2)

with open(save_user_aspect_total, "w", encoding="utf-8") as f:
    json.dump(user_aspect_total, f, ensure_ascii=False, indent=2)

print(f"Saved sentiment dict → {save_user_aspect_sent}")
print(f"Saved total dict → {save_user_aspect_total}")
