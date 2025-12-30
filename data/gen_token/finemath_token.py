import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from itertools import chain
from datasets import logging as ds_logging
from datasets import Features, Value

# --- 1. 配置参数 ---
DATA_FILES_PATH = "/home/users/nus/e1352689/scratch/finemath/finemath-4plus/"
TOKENIZER_PATH = "/home/users/nus/e1352689/pretrain-LLM-from-scratch/tokenizer"
PROCESSED_DATA_PATH = "/home/users/nus/e1352689/scratch/finemath_processed"
BLOCK_SIZE = 3072
NUM_PROC = os.cpu_count() // 2 or 8

def prepare_and_tokenize_dataset():
    """
    加载、筛选、编码并块化数据集（仅保存 input_ids）。
    """

    print("=== 步骤 1/4: 收集 Parquet 文件 ===")
    data_files = [os.path.join(DATA_FILES_PATH, f) for f in os.listdir(DATA_FILES_PATH) if f.endswith(".parquet")]
    # data_files = [DATA_FILES_PATH + "zh-001.parquet"]
    if not data_files:
        raise ValueError(f"在目录 {DATA_FILES_PATH} 中没有找到 Parquet 文件")
    print(f"找到 {len(data_files)} 个 Parquet 文件")

    print("\n=== 步骤 2/4: 加载数据集（content/score/source） ===")
    # 不使用 streaming 以便多进程 map/filter，并能获得长度与进度
    raw_ds = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        streaming=False,
        features=Features({
            "content": Value("string"),
            "score": Value("float32"),
            "source": Value("string"),
        }),
    )
    # 仅提示一个示例（如果有）
    try:
        ex = next(iter(raw_ds))
        print(f"示例行：content[:80]={ex.get('content','')[:80]!r}, score={ex.get('score')}, source={ex.get('source')}")
    except StopIteration:
        raise ValueError("数据集为空，请检查 Parquet 文件内容")

    # 确保存在所需列
    required_cols = {"text"}
    missing = required_cols - set(raw_ds.column_names)
    if missing:
        raise ValueError(f"缺失必须列: {missing}，现有列: {raw_ds.column_names}")
    
    filtered_ds = raw_ds

    print("\n=== 步骤 3/4: 加载 Tokenizer 并进行分词 ===")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer 无 pad_token，将其设置为 eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # 解析分隔符 token：优先使用 literal '</s>'，否则回退到 eos_token
    SEP_TEXT = "</s>"
    sep_ids = tokenizer.encode(SEP_TEXT, add_special_tokens=False)
    if not sep_ids:
        # 当词表中没有 '</s>' 时，回退为 eos
        sep_ids = [tokenizer.eos_token_id]
        print("警告：词表中无 '</s>'，将使用 eos_token 作为分隔。")
    SEP_IDS = sep_ids  # list[int]

    def tokenize_function(examples):
        # 在每条样本末尾追加分隔符文本，以保证拼接时文档间有边界
        texts = [(t if isinstance(t, str) else "") + SEP_TEXT for t in examples["content"]]
        # 仅返回 input_ids，不生成 attention_mask 以节省空间
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return {"input_ids": encoded["input_ids"]}

    tokenized_ds = filtered_ds.map(
        tokenize_function,
        batched=True,
        batch_size=NUM_PROC * 5,
        num_proc=NUM_PROC // 2,
        remove_columns=filtered_ds.column_names,  # 丢弃原始列，仅保留我们映射出的列
        desc="Tokenizing (with </s> separators)",
    )
    print(f"分词完成：样本数 = {len(tokenized_ds)}；列 = {tokenized_ds.column_names}")

    print("\n=== 步骤 5/5: 块化为固定长度（仅保留 input_ids） ===")
    def group_texts(examples):
        # 仅有 input_ids 一列
        concatenated = list(chain.from_iterable(examples["input_ids"]))
        total_length = len(concatenated)
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # 切分为定长块
        result_input_ids = [
            concatenated[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)
        ]
        return {"input_ids": result_input_ids}


    lm_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=NUM_PROC * 10,
        num_proc=NUM_PROC,
        desc=f"Grouping into blocks of {BLOCK_SIZE}",
    )

    final_count = len(lm_ds)
    print(f"块化完成！最终样本数（block 数）：{final_count}")

    print(f"\n=== 保存到磁盘：{PROCESSED_DATA_PATH} ===")
    # 保存时数据集只含 input_ids 列
    lm_ds.save_to_disk(PROCESSED_DATA_PATH)
    print("✅ 所有数据准备工作已完成（仅保存 input_ids）。")

if __name__ == "__main__":
    prepare_and_tokenize_dataset()
