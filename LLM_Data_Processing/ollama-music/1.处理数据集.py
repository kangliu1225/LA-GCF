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
    # 去掉代码块标记
    text = re.sub(r"```json|```", "", text)
    # 去掉花括号和多余引号
    # text = text.strip("{} \n\t").replace('"', '')
    # 用逗号加空格分隔
    # text = re.sub(r"\s*,\s*", ", ", text)
    # 去掉内部换行
    text = re.sub(r"\s*\n\s*", ", ", text)
    return text.strip()

def main():
    # 读取输入文件
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, sentence in enumerate(tqdm(lines, desc="Processing reviews")):
            prompt = f"\nreviews:\n{sentence}\n"
            try:
                start = time.time()
                response = ollama.chat(
                    model="gemma3:27b-it-qat",
                    messages=[{"role": "user", "content": prompt_pro + prompt}],
                )
                raw_output = response["message"]["content"]
                clean_text = clean_output(raw_output)

                # 每条结果写一行
                f_out.write(clean_text + "\n")
                f_out.flush()

            except Exception as e:
                tqdm.write(f"[错误] 第 {i+1} 行处理失败: {e}")
                f_out.write("ERROR\n")
                f_out.flush()
                time.sleep(2)

if __name__ == "__main__":
    main()
