import torch
import json
from tqdm import tqdm

# 1. 读取pkl
file_path = r"D:\workspace\dataset\aspect\yelp\aspect_embeddings_no_whiten.pkl"
data = torch.load(file_path, map_location='cpu')  # dict: {aspect: tensor}

# 2. 准备向量和对应的aspect列表
aspects = list(data.keys())
vectors = torch.stack([data[a] for a in aspects])  # shape: [N, dim]

# 3. 归一化向量（计算余弦相似度前必须归一化）
vectors = vectors / vectors.norm(dim=1, keepdim=True)

# 4. 计算相似度矩阵（N x N）
# 注意：矩阵较大时占用内存 ~N^2，如果N=10000, 约 10000*10000*4byte=400MB
sim_matrix = vectors @ vectors.T  # 余弦相似度矩阵

# 5. 构建相似字典
threshold = 0.88      #相似度阈值
similar_dict = {}

for i in tqdm(range(len(aspects)), desc="Building similarity dict"):
    # 找出相似度大于阈值的索引
    sim_idx = (sim_matrix[i] > threshold).nonzero(as_tuple=True)[0].tolist()
    sim_idx = [j for j in sim_idx if j != i]  # 排除自身
    similar_aspects = [aspects[j] for j in sim_idx]
    # if aspects[i] == "song" and "songs" in similar_aspects:
    #     cos_sim = sim_matrix[i, aspects.index("songs")].item()
    #     print(f"'song' and 'songs' similarity: {cos_sim:.4f}")
    similar_dict[aspects[i]] = similar_aspects

# 6. 保存为JSON
output_path = rf"D:\workspace\dataset\aspect\yelp\8.aspect_similarity_{threshold}_no_whiten.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(similar_dict, f, ensure_ascii=False, indent=2)

print(f"Saved {len(similar_dict)} aspects similarity dict to {output_path}")
