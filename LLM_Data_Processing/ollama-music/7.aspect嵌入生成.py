# -*- coding: utf-8 -*-
"""
Extract BERT-Whitening embeddings for aspects
Reference: https://kexue.fm/archives/8069
"""
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
import json
import os

# ===== 配置 =====
input_path = r"D:\workspace\dataset\aspect\yelp\aspect_counts.json"
save_path = r"D:\workspace\dataset\aspect\yelp\aspect_embeddings_no_whiten.pkl"
model_name = "bert-base-uncased"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
vec_dim = 64

# ===== 加载模型 =====
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name).to(device)
bert.config.output_hidden_states = True

# ===== 读取aspect =====
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# aspect_counts.json 格式: {"(music, positive)": 1, "(songs, positive)": 1, ...}
aspects = list(data.keys())
print(f"Loaded {len(aspects)} aspects.")

# ===== whitening工具函数 =====
def compute_kernel_bias(vecs, dim):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, _ = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :dim], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    if kernel is not None and bias is not None:
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

# ===== 提取embedding =====
all_vecs = []
batch_size = 32

for i in tqdm(range(0, len(aspects), batch_size), desc="Extracting aspect embeddings"):
    batch_aspects = aspects[i:i+batch_size]
    encoding = tokenizer(batch_aspects, return_tensors="pt", padding=True, truncation=True, max_length=32)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert(input_ids, attention_mask)
        hidden_states = outputs.hidden_states
        last2 = (hidden_states[-1] + hidden_states[-2]) / 2
        sentence_vecs = (attention_mask.unsqueeze(-1) * last2).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        all_vecs.append(sentence_vecs.cpu().numpy())

all_vecs = np.vstack(all_vecs)
print(f"Raw embeddings shape: {all_vecs.shape}")

# ===== whitening + normalize =====
# kernel, bias = compute_kernel_bias(all_vecs, vec_dim)
# vecs_final = transform_and_normalize(all_vecs, kernel, bias)

# ===== 保存结果 =====
aspect_emb = {aspect: torch.tensor(all_vecs[i]) for i, aspect in enumerate(aspects)}
torch.save(aspect_emb, save_path)

print(f"✅ Saved {len(aspect_emb)} aspect embeddings to {save_path}")
