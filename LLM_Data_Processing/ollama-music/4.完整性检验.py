from tqdm import tqdm
import json

output_path = r"D:/workspace/dataset/aspect/yelp/1.train_pred_gemma3.txt"

def is_valid_format_split(line: str) -> bool:
    """
    检查一行是否符合 (aspect, sentiment): count, (aspect, sentiment): count 这种格式
    aspect 可以是多个单词
    """

    parts=json.loads(line)
    for key,value in parts.items():
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


def main():
    total = 0
    error_count = 0
    none_count = 0
    invalid_count = 0

    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Checking lines")):
            total += 1
            text = line.strip()

            if text.upper() == "ERROR":
                error_count += 1
            elif text.lower() == "none":
                none_count += 1
            elif not is_valid_format_split(text):
                invalid_count += 1
                print(f"[❌ 格式错误] 第 {i+1} 行: {text}")

    print("\n========== 检查结果 ==========")
    print(f"总行数: {total}")
    print(f"ERROR 行数: {error_count}")
    print(f"none 行数: {none_count}, 占比：{(none_count/total)* 100:.5f}%")
    print(f"格式错误行数: {invalid_count}")
    print(f"通过率: {(1 - invalid_count/total) * 100:.5f}%")
    print("==============================")

if __name__ == "__main__":
    main()
