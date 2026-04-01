import ollama
import time
import re
from tqdm import tqdm

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
{"(battery, negative)": 1, "(battery, positive)": 1, "(screen, positive)": 2}.
or 
none
"""

def clean_output(raw_output: str) -> str:
    """
    清理模型输出，保证每条结果为一行：
    (battery, negative): 1, (screen, positive): 2
    """
    text = raw_output.strip()
    text = re.sub(r"```json|```", "", text)
    text = text.strip("{} \n\t").replace('"', '')
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*\n\s*", ", ", text)
    return text.strip()

def main():
    # 读取原始输入和已处理文件
    with open(input_path, "r", encoding="utf-8") as f_in:
        reviews = [line.strip() for line in f_in if line.strip()]

    with open(output_path, "r", encoding="utf-8") as f_out:
        results = [line.rstrip("\n") for line in f_out]

    if len(results) != len(reviews):
        print(f"⚠️ 警告：输入({len(reviews)})与输出({len(results)})行数不一致，请检查文件。")
        return

    error_indices = [i for i, line in enumerate(results) if line.strip().upper() == "ERROR"]
    if not error_indices:
        print("✅ 没有检测到 ERROR 行，文件已全部正确处理。")
        return

    print(f"🔄 检测到 {len(error_indices)} 条 ERROR 行，开始重新处理...")

    for idx in tqdm(error_indices, desc="Reprocessing ERROR lines"):
        sentence = reviews[idx]
        prompt = f"\nreviews:\n{sentence}\n"

        try:
            start = time.time()
            response = ollama.chat(
                model="gemma3:27b-it-qat",
                messages=[{"role": "user", "content": prompt_pro + prompt}],
            )
            raw_output = response["message"]["content"]
            clean_text = clean_output(raw_output)
            elapsed = time.time() - start

            results[idx] = clean_text
            tqdm.write(f"[修复] 第 {idx+1} 行成功 ({elapsed:.2f}s)")
            time.sleep(0.5)

        except Exception as e:
            tqdm.write(f"[错误] 第 {idx+1} 行重试失败: {e}")
            time.sleep(2)

    # 重新写回完整文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in results:
            f_out.write(line + "\n")

    print(f"✅ 修复完成，共修复 {len(error_indices)} 条 ERROR 行。")

if __name__ == "__main__":
    main()
