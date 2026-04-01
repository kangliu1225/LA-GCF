import ollama
import time
import re
from tqdm import tqdm
import json
input_path = r"D:/workspace/dataset/aspect/yelp/wait_pre.txt"
output_path = r"D:/workspace/dataset/aspect/yelp/1.train_pred_gemma3.txt"

prompt_pro = """
Act as an aspect-based sentiment analysis expert for E-commerce platform. 
Given the following review written by a user on various products, extract all aspects mentioned in the text. 
Include implicit aspects. For each aspect, determine the sentiment polarity as positive, negative, or neutral. 
Additionally, provide counts of aspects grouped by their sentiment polarity (positive, negative, neutral). 
Provide the output directly in a structured JSON format, but do not include code block markers, do not add line breaks, and do not use outer braces. 
Each aspect–sentiment pair should be used as a key, and the corresponding count should be the value.
If the review contains no aspect related to user preferences, both the key and value should be "none".
Rules:
1. Only extract actual aspects (usually nouns representing product features or items). 
   Do NOT treat sentiment words (like "love", "great", "good", "bad") as aspects.
2. Output format must strictly follow this example:
{"(battery, negative)": "1", "(battery, positive)": "1", "(screen, positive)": "2"}
or 
none
"""

def clean_output(raw_output: str) -> str:
    """
    清理模型输出，保证每条结果为一行：
    (battery, negative): 1, (screen, positive): 2
    """
    text = raw_output.strip()
    # 去掉代码块标记
    text = re.sub(r"```json|```", "", text)
    # 去掉花括号和多余引号
    # text = text.strip("{} \n\t").replace('"', '')
    # 用逗号加空格分隔
    # text = re.sub(r"\s*,\s*", ", ", text)
    # 去掉内部换行
    text = re.sub(r"\s*\n\s*", ", ", text)
    if text=='{"none": "none"}' or text=='none, </end_of_turn>':
        return "none"
    return text.strip()

def is_valid_format(line: str) -> bool:
    """
    检查是否符合 (aspect, sentiment): count 格式，
    允许多个以逗号分隔的项。
    """
    if line == 'none':
        return  True
    if not line or line.upper() == "ERROR":
        return False
    if line=='{"none": "none"}':
        a=1
    try:
        parts = json.loads(line)
        for key, value in parts.items():
            key = key.strip()

            # 检查数字
            if int(value) < 0:
                return False

            # 检查左侧格式 "(aspect, sentiment)"
            if not (key.startswith("(") and key.endswith(")")):
                return False
            content = key[1:-1]  # 去掉括号
            if "," not in content:
                return False
            aspect, sentiment = [c.strip() for c in content.split(",", 1)]
            if not aspect:  # aspect不能为空
                return False
            if sentiment not in ("positive", "negative", "neutral"):
                return False
        return True
    except:
        return False

    # pattern = r'^\(([^,]+),\s*(positive|negative|neutral)\):\s*\d+(,\s*\(([^,]+),\s*(positive|negative|neutral)\):\s*\d+)*$'
    # return re.match(pattern, line.strip()) is not None

def regenerate_line(sentence: str) -> str:
    """重新生成一行结果，若仍无效则返回 'none'"""
    prompt = f"\nreviews:\n{sentence}\n"
    try:
        response = ollama.chat(
            model="gemma3:27b-it-qat",
            messages=[{"role": "user", "content": prompt_pro + prompt}],
        )
        raw_output = response["message"]["content"]
        clean_text = clean_output(raw_output)
        if is_valid_format(clean_text):
            return clean_text
        else:
            print(f"[⚠️ 格式仍错误] → {clean_text}")
            return "none"
    except Exception as e:
        print(f"[错误] 重新处理失败: {e}")
        return "none"

def main():
    # 读取输入评论与输出结果
    with open(input_path, "r", encoding="utf-8") as f_in:
        reviews = [line.strip() for line in f_in if line.strip()]

    with open(output_path, "r", encoding="utf-8") as f_out:
        results = [line.rstrip("\n") for line in f_out]

    if len(results) != len(reviews):
        print(f"⚠️ 输入({len(reviews)})与输出({len(results)})行数不一致，请检查文件。")
        return

    print("🔍 开始格式检测与自动修复...")

    fixed_count = 0
    for i, (review, result) in enumerate(tqdm(zip(reviews, results), total=len(results), desc="Checking")):
        if not is_valid_format(result):
            tqdm.write(f"[❌ 格式错误] 第 {i+1} 行: {result}")
            new_result = regenerate_line(review)
            results[i] = new_result
            fixed_count += 1
            time.sleep(0.5)

    # 写回文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in results:
            f_out.write(line + "\n")

    print(f"✅ 检测完成，共修复 {fixed_count} 行。")

if __name__ == "__main__":
    main()
