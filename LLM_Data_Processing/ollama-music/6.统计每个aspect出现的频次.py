import json
from collections import defaultdict

input_path = r"D:/workspace/dataset/aspect/yelp/1.train_pred_gemma3.txt"
output_path = r"D:/workspace/dataset/aspect/yelp/aspect_counts.json"


# 存储每个 aspect 总出现次数
aspect_counts = defaultdict(int)

# 逐行读取文件
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.lower() == "none" or not line:
            continue
        try:
            # 将每行字符串解析为字典
            data = json.loads(line)
            for key, count in data.items():
                # key 形如 "(aspect, sentiment)"
                aspect_name = key.split(",")[0].strip("() ")
                aspect_counts[aspect_name] += int(count)
        except json.JSONDecodeError:
            print(f"[格式错误] 跳过这一行: {line}")

# 输出统计结果
print("每个 aspect 总出现次数:")
for aspect, count in aspect_counts.items():
    print(f"{aspect}: {count}")

print("\n不同 aspect 种类数:", len(aspect_counts))

# 保存为 JSON 文件
with open(output_path, "w", encoding="utf-8") as f_out:
    json.dump(aspect_counts, f_out, ensure_ascii=False, indent=2)

print(f"统计完成，已保存到 {output_path}")