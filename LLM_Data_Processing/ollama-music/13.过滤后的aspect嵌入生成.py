# -*- coding: utf-8 -*-
"""
Extract BERT embeddings for aspects (no whitening)
Reference: https://kexue.fm/archives/8069
"""

import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
import json
import os

# ===== 配置 =====
input_path = r"D:\workspace\dataset\aspect\dm5\12.user_aspect_total.json"
save_path = r"D:\workspace\dataset\aspect\dm5\13.aspect_embeddings_from_total.pkl"
model_name = "bert-base-uncased"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
vec_dim = 64#768  # 原始BERT维度

# ===== 加载模型 =====
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name).to(device)
bert.config.output_hidden_states = True
bert.eval()

# ===== 读取所有aspect =====
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

aspect_set = set()
for user, aspects in data.items():
    aspect_set.update(aspects.keys())

aspects = sorted(list(aspect_set))
print(f"Loaded {len(aspects)} unique aspects.")

# ===== 提取embedding =====
all_vecs = []
batch_size = 32

for i in tqdm(range(0, len(aspects), batch_size), desc="Extracting aspect embeddings"):
    batch_aspects = aspects[i:i + batch_size]
    encoding = tokenizer(batch_aspects, return_tensors="pt", padding=True, truncation=True, max_length=32)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert(input_ids, attention_mask)
        hidden_states = outputs.hidden_states
        # 使用最后两层平均
        last2 = (hidden_states[-1] + hidden_states[-2]) / 2
        # mask 平均池化
        sentence_vecs = (attention_mask.unsqueeze(-1) * last2).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        all_vecs.append(sentence_vecs.cpu().numpy())

all_vecs = np.vstack(all_vecs)
print(f"Raw embeddings shape: {all_vecs.shape}")

# ===== 保存结果 =====
aspect_emb = {aspect: torch.tensor(all_vecs[i])[:vec_dim] for i, aspect in enumerate(aspects)}
torch.save(aspect_emb, save_path)

print(f"Saved {len(aspect_emb)} aspect embeddings to {save_path}")
