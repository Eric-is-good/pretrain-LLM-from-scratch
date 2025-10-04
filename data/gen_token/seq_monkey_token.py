import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from itertools import chain
from datasets import logging as ds_logging
from datasets import Features, Value

# --- 1. 配置参数 ---
# !!! 请将这里修改为您 JSONL 文件的实际路径 !!!
DATA_FILE_PATH = "/home/users/nus/e1352533/mydata/data2/sql_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl"
TOKENIZER_PATH = "/home/users/nus/e1352533/code/pretrain-LLM-from-scratch/tokenizer"
PROCESSED_DATA_PATH = "/home/users/nus/e1352533/mydata/data2/sql_monkey/mobvoi_seq_monkey_general_open_corpus/" # 建议为新数据指定一个新输出目录
BLOCK_SIZE = 4096
# 根据您的机器配置调整进程数
NUM_PROC = os.cpu_count() // 4 if os.cpu_count() else 8

def prepare_and_tokenize_dataset_from_jsonl():
    """
    加载、筛选、编码并块化一个 JSONL 数据集（仅保存 input_ids）。
    """

    print("=== 步骤 1/5: 检查 JSONL 文件路径 ===")
    if not os.path.exists(DATA_FILE_PATH) or not DATA_FILE_PATH.endswith(".jsonl"):
        raise ValueError(f"文件路径 {DATA_FILE_PATH} 不存在或不是一个 .jsonl 文件")
    print(f"准备加载文件: {DATA_FILE_PATH}")


    print("\n=== 步骤 2/5: 加载 JSONL 数据集 (text) ===")
    # 加载 JSONL 文件，每行是一个 JSON 对象
    raw_ds = load_dataset(
        "json",
        data_files=DATA_FILE_PATH,
        split="train", # 对于单个文件，通常 split 为 "train"
        streaming=False, # 保持 False 以便多进程处理和获取长度
    )
    
    # 确保存在 'text' 列
    if "text" not in raw_ds.column_names:
        raise ValueError(f"JSONL 文件中必须包含 'text' 键。找到的列: {raw_ds.column_names}")

    # 打印一个示例
    try:
        ex = next(iter(raw_ds))
        print(f"示例行: text[:80]={ex.get('text', '')[:80]!r}")
    except StopIteration:
        raise ValueError("数据集为空，请检查 JSONL 文件内容")


    print("\n=== 步骤 3/5: 筛选空样本 ===")
    before = len(raw_ds)
    # 筛选 text 字段是有效字符串的样本
    filtered_ds = raw_ds.filter(
        lambda x: isinstance(x["text"], str) and len(x["text"]) > 0,
        num_proc=NUM_PROC,
        desc="Filtering out empty samples",
    )
    after = len(filtered_ds)
    if after == 0:
        raise ValueError("筛选后无有效样本。请检查数据内容。")
    print(f"筛选完成：{before} -> {after} 条")


    print("\n=== 步骤 4/5: 加载 Tokenizer 并进行分词 ===")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer 无 pad_token，将其设置为 eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # 文档分隔符，用于在拼接不同文档时提供边界
    SEP_TEXT = "</s>"

    def tokenize_function(examples):
        # 从 "text" 列获取文本，并在每条末尾追加分隔符
        texts_with_sep = [t + SEP_TEXT for t in examples["text"]]
        # 仅返回 input_ids，不生成 attention_mask 以节省空间和内存
        encoded = tokenizer(
            texts_with_sep,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return {"input_ids": encoded["input_ids"]}

    tokenized_ds = filtered_ds.map(
        tokenize_function,
        batched=True,
        batch_size=1024, # 可以根据内存调整
        num_proc=NUM_PROC,
        remove_columns=filtered_ds.column_names,  # 移除原始 "text" 列
        desc="Tokenizing (with </s> separators)",
    )
    print(f"分词完成：样本数 = {len(tokenized_ds)}；列 = {tokenized_ds.column_names}")


    print("\n=== 步骤 5/5: 块化为固定长度 (Grouping) ===")
    def group_texts(examples):
        # 将多个样本的 input_ids 列表拼接成一个长列表
        concatenated_ids = list(chain.from_iterable(examples["input_ids"]))
        total_length = len(concatenated_ids)
        
        # 我们只保留完整的块，丢弃末尾不足 BLOCK_SIZE 的部分
        # total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # 上面的写法如果total_length < BLOCK_SIZE, 结果会是0。
        # 以下写法更安全，如果 total_length < BLOCK_SIZE, 会产生一个空列表，不会报错
        if total_length < BLOCK_SIZE:
             return {"input_ids": []} # 返回空，map会自动处理
        
        # 计算可以形成多少个完整的块
        num_blocks = total_length // BLOCK_SIZE
        # 截断，只保留能被 BLOCK_SIZE 整除的部分
        truncated_length = num_blocks * BLOCK_SIZE
        
        # 切分为定长块
        result = [
            concatenated_ids[i : i + BLOCK_SIZE] for i in range(0, truncated_length, BLOCK_SIZE)
        ]
        return {"input_ids": result}


    lm_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=2048, # 增大批大小以提高效率
        num_proc=NUM_PROC,
        desc=f"Grouping into blocks of {BLOCK_SIZE}",
    )

    final_count = len(lm_ds)
    if final_count == 0:
        print("警告：最终没有生成任何数据块。可能是因为总 token 数小于 BLOCK_SIZE。")
    else:
        print(f"块化完成！最终样本数（block 数）：{final_count}")


    print(f"\n=== 保存到磁盘：{PROCESSED_DATA_PATH} ===")
    # 确保输出目录存在
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    # 保存时数据集只含 input_ids 列
    lm_ds.save_to_disk(PROCESSED_DATA_PATH)
    print("✅ 所有数据准备工作已完成。")


if __name__ == "__main__":
    prepare_and_tokenize_dataset_from_jsonl()