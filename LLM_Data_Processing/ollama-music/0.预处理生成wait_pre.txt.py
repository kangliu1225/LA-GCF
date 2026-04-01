import json

input_path = r"D:\workspace\Review_Yelp\data\yelp2013\yelp2013_sentiment_train.tsv"
output_path = r"D:\workspace\dataset\aspect\yelp\wait_pre.txt"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        # 每一行是一个 JSON 字符串
        data = json.loads(line)
        review_text = data.get("review_text", "").strip()
        if review_text:
            fout.write(review_text + "\n")

print(f"✅ 已保存到 {output_path}")
