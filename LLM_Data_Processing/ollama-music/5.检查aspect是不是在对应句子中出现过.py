import re
import json
from tqdm import tqdm

# 输入文件：原句
input_path = r"D:/workspace/dataset/aspect/yelp/wait_pre.txt"
# 处理结果文件：每行一个处理结果
result_path = r"D:/workspace/dataset/aspect/yelp/1.train_pred_gemma3.txt"

missing_count = 0
total_count = 0

with open(input_path, "r", encoding="utf-8") as f_in, \
     open(result_path, "r", encoding="utf-8") as f_res:

    for line_num, (sentence, result_line) in enumerate(tqdm(zip(f_in, f_res), desc="Checking aspects"), start=1):
        total_count += 1
        sentence = sentence.strip().lower()
        result_line = result_line.strip()

        # 跳过 none 行
        if result_line.lower() == "none":
            continue

        try:
            # 加上花括号解析为字典
            data = json.loads(result_line)
        except json.JSONDecodeError:
            print(f"[JSON解析错误] 第 {line_num} 行: {result_line}")
            continue

        # 检查每个 aspect 是否在句子中出现
        missing_aspects = []
        for key in data.keys():
            # key 形如 "(aspect, sentiment)"
            match = re.match(r"\((.+?),\s*(positive|negative|neutral)\)", key)
            if match:
                aspect = match.group(1).lower()
                # 简单判断 aspect 是否出现在句子中
                if aspect not in sentence:
                    missing_aspects.append(aspect)
            else:
                missing_aspects.append(key)

        if missing_aspects:
            missing_count += 1
            print(f"[未出现] 第 {line_num} 行: {sentence}")
            print(f"提取结果: {result_line}\n")



print("========== 检查统计 ==========")
print(f"总句子数: {total_count}")
print(f"包含未出现 aspect 的行数: {missing_count}")
print("==============================")
