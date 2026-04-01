import json

# 文件路径
data_path = r"D:\workspace\Review_Music\data\Digital_Music_5\Digital_Music_5_sentiment_train.tsv"
aspect_path = r"D:\workspace\dataset\aspect\dm5\11.train_pred_gemma3_filtered.txt"

user_item_aspect = {}

with open(data_path, 'r', encoding='utf-8') as f1, open(aspect_path, 'r', encoding='utf-8') as f2:
    for review_line, aspect_line in zip(f1, f2):
        review_data = json.loads(review_line.strip())
        user = review_data["user_id"]
        item = review_data["item_id"]

        key = f"{user},{item}"

        # 跳过 "none" 的行
        if aspect_line.strip() == "none":
            continue

        # 解析aspect预测结果
        aspect_data = json.loads(aspect_line.strip())
        pairs = []
        for k, v in aspect_data.items():
            aspect, sentiment = k.strip("()").split(", ")
            # v 表示出现次数，重复添加多次
            pairs.extend([(aspect, sentiment)] * v)

        user_item_aspect[key] = pairs

# 输出部分结果看看
for k, v in list(user_item_aspect.items())[:5]:
    print(k, v)

# 可选：保存结果
out_path = r"D:\workspace\dataset\aspect\dm5\12.user_item_aspect_graph.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(user_item_aspect, f, ensure_ascii=False, indent=2)

print(f"保存完成：{out_path}")
