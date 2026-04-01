import json

# ==== 文件路径 ====
pred_path = r"D:\workspace\dataset\aspect\dm5\1.train_pred_gemma3.txt"
group_path = r"D:\workspace\dataset\aspect\dm5\10.top_aspect_groups_no_whiten.txt"
output_path = r"D:\workspace\dataset\aspect\dm5\11.train_pred_gemma3_filtered.txt"

# ==== 1. 读取 aspect 分组 ====
aspect_map = {}  # aspect -> group代表词
with open(group_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        group_rep = parts[0]
        for p in parts:
            aspect_map[p] = group_rep

print(f"Loaded {len(aspect_map)} aspects from groups")

# ==== 2. 读取预测文件 ====
new_lines = []
with open(pred_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line == "none" or not line:
            new_lines.append("none")
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append("none")
            continue

        new_record = {}
        for k, v in record.items():
            # 解析 "(aspect, sentiment)" 结构
            if not (k.startswith("(") and "," in k):
                continue
            aspect = k.split(",")[0].strip(" (")
            sentiment = k.split(",")[1].strip(" )")

            # 过滤：只保留在 aspect_map 中的 aspect
            if aspect not in aspect_map:
                continue

            # 替换为代表词
            group_aspect = aspect_map[aspect]
            new_key = f"({group_aspect}, {sentiment})"

            # 累加计数
            new_record[new_key] = new_record.get(new_key, 0) + int(v)

        if new_record:
            new_lines.append(json.dumps(new_record, ensure_ascii=False))
        else:
            new_lines.append("none")

# ==== 3. 保存结果 ====
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))

print(f"Saved filtered file to {output_path}")
