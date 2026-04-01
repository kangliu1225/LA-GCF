import json

# 文件路径
counts_path = r"D:\workspace\dataset\aspect\dm5\aspect_counts.json"
sim_path = r"D:\workspace\dataset\aspect\dm5\8.aspect_similarity_0.88_no_whiten.json"
output_path = r"D:\workspace\dataset\aspect\dm5\10.top_aspect_groups_no_whiten.txt"

# 读取数据
with open(counts_path, "r", encoding="utf-8") as f:
    aspect_counts = json.load(f)

with open(sim_path, "r", encoding="utf-8") as f:
    aspect_sim = json.load(f)

# 统计新的计数（自己+相似aspect）
new_counts = {}
for aspect in aspect_counts:
    total = aspect_counts.get(aspect, 0)
    for sim_aspect in aspect_sim.get(aspect, []):
        total += aspect_counts.get(sim_aspect, 0)
    new_counts[aspect] = total

# 按计数排序
sorted_aspects = sorted(new_counts.items(), key=lambda x: x[1], reverse=True)

seen = set()  # 去重记录
groups = []

for aspect, _ in sorted_aspects:
    if aspect in seen:
        continue
    group = [aspect]
    for sim_aspect in aspect_sim.get(aspect, []):
        if sim_aspect not in seen:
            group.append(sim_aspect)
    # 标记所有加入的aspect为已处理
    for a in group:
        seen.add(a)
    groups.append(group)
    if len(groups) >= 2048:
        break

# 对每一行的aspect按照原数量从大到小排序 D:10.14
for idx, group in enumerate(groups):
    groups[idx]=sorted(group,key=lambda x: aspect_counts[x], reverse=True)




# 保存为txt，每行一个列表，用逗号连接
with open(output_path, "w", encoding="utf-8") as f:
    for group in groups:
        f.write(",".join(group) + "\n")

print(f"Saved top aspect groups to {output_path}")
