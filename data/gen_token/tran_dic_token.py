import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from itertools import chain
import glob

# --- 1. 配置参数 (已更新) ---
# 修改为您的主文件夹路径
DATA_DIR_PATH = "/home/users/nus/e1352533/mydata/trans_dic/DICT"  # <--- 请将这里替换为您的 DICT 文件夹实际路径
TOKENIZER_PATH = "/home/users/nus/e1352533/code/pretrain-LLM-from-scratch/tokenizer"
# 建议为不同数据集使用不同输出目录
PROCESSED_DATA_PATH = "/home/users/nus/e1352533/mydata/trans_dic/DICT_processed_data"
BLOCK_SIZE = 4096
NUM_PROC = os.cpu_count() // 2 or 8
# 关键：指定文件的原始编码，解决乱码问题。'gbk' 是最常见的中文 Windows 编码。
# 如果 'gbk' 仍然无效，可以尝试 'gb2312' 或 'gb18030'。
SOURCE_FILE_ENCODING = 'gb18030'

def prepare_and_tokenize_dataset():
    """
    加载、编码并块化文件夹中的所有 TXT 文件（仅保存 input_ids）。
    (新版本：使用自定义生成器以忽略解码错误)
    """

    print("=== 步骤 1/5: 收集 TXT 文件 ===")
    # 使用 glob 递归查找所有 txt 文件
    file_paths = glob.glob(os.path.join(DATA_DIR_PATH, "**/*.txt"), recursive=True)
    if not file_paths:
        raise ValueError(f"在目录 {DATA_DIR_PATH} 中没有找到 .txt 文件")
    print(f"找到 {len(file_paths)} 个 TXT 文件")

    # 自定义数据加载生成器
    def text_file_generator():
        for path in file_paths:
            try:
                # 使用 errors='ignore' 来跳过无法解码的字节
                with open(path, 'r', encoding=SOURCE_FILE_ENCODING, errors='ignore') as f:
                    for line in f:
                        # yield 一个字典，这是 datasets 库期望的格式
                        yield {"text": line}
            except Exception as e:
                print(f"警告：读取文件 {path} 时出错: {e}")

    print(f"\n=== 步骤 2/5: 加载数据集 (使用自定义生成器，忽略编码错误) ===")
    # 从我们的生成器创建数据集
    raw_ds = Dataset.from_generator(text_file_generator)

    if len(raw_ds) == 0:
        raise ValueError(f"在目录 {DATA_DIR_PATH} 中没有加载到任何内容。")
    
    print(f"原始数据集加载完成，共 {len(raw_ds)} 行文本。")
    print(f"列名: {raw_ds.column_names}")
    
    # 后续步骤与之前完全相同
    print("\n=== 步骤 3/5: 筛选空行或过短的行 ===")
    # ... (这部分代码和之前一样，这里省略以便简洁)
    before = len(raw_ds)
    filtered_ds = raw_ds.filter(
        lambda x: x["text"] and len(x["text"].strip()) > 0,
        num_proc=NUM_PROC,
        desc="Filtering out empty or very short lines",
    )
    after = len(filtered_ds)
    if after == 0:
        raise ValueError("筛选后无有效样本。")
    print(f"筛选完成：{before} -> {after} 行")


    print("\n=== 步骤 4/5: 加载 Tokenizer 并进行分词 ===")
    # ... (这部分代码和之前一样，这里省略以便简洁)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=False,
        )
    tokenized_ds = filtered_ds.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        remove_columns=filtered_ds.column_names,
        desc="Tokenizing lines",
    )

    print("\n=== 步骤 5/5: 块化为固定长度（仅保留 input_ids） ===")
    # ... (这部分代码和之前一样，这里省略以便简洁)
    def group_texts(examples):
        concatenated = list(chain.from_iterable(examples["input_ids"]))
        total_length = len(concatenated)
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result_input_ids = [
            concatenated[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)
        ]
        return {"input_ids": result_input_ids}
    lm_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=2000,
        num_proc=NUM_PROC,
        desc=f"Grouping into blocks of {BLOCK_SIZE}",
    )
    
    final_count = len(lm_ds)
    print(f"块化完成！最终样本数（block 数）：{final_count}")
    print(f"\n=== 保存到磁盘：{PROCESSED_DATA_PATH} ===")
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    lm_ds.save_to_disk(PROCESSED_DATA_PATH)
    print("✅ 所有数据准备工作已完成（仅保存 input_ids）。")

if __name__ == "__main__":
    prepare_and_tokenize_dataset()