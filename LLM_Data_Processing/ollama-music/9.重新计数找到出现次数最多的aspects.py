import json
from collections import defaultdict

# 文件路径
counts_path = r"D:\workspace\dataset\aspect\dm5\aspect_counts.json"
sim_path = r"D:\workspace\dataset\aspect\dm5\8.aspect_similarity_0.88_no_whiten.json"
output_path = r"D:\workspace\dataset\aspect\dm5\9.aspect_counts_aggregated_no_whiten.json"

# 1. 读取原始数据
with open(counts_path, "r", encoding="utf-8") as f:
    aspect_counts = json.load(f)  # {"song": 60614, ...}

with open(sim_path, "r", encoding="utf-8") as f:
    aspect_sim = json.load(f)  # {"song": ["songs", "music", ...], ...}

# 2. 统计新的计数
new_counts = {}

for aspect in aspect_counts:
    total = aspect_counts.get(aspect, 0)
    # 加上相似 aspect 的计数
    for sim_aspect in aspect_sim.get(aspect, []):
        total += aspect_counts.get(sim_aspect, 0)
    new_counts[aspect] = total

# 3. 去重处理
# 如果 A 已经在 B 的相似集合里，则只保留一个出现
final_counts = {}
seen = set()

for aspect, count in sorted(new_counts.items(), key=lambda x: x[1], reverse=True):
    if aspect in seen:
        continue
    # 将当前aspect和其相似的aspect都标记为已处理
    seen.add(aspect)
    for sim_aspect in aspect_sim.get(aspect, []):
        seen.add(sim_aspect)
    final_counts[aspect] = count

# 4. 获取出现次数最多的前1000个aspect
top_100 = dict(sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:2048])

# 5. 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(top_100, f, ensure_ascii=False, indent=2)

print(f"Saved top 512 aggregated aspect counts to {output_path}")
