import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from itertools import chain
from datasets import logging as ds_logging
from datasets import Features, Value

# --- 1. 配置参数 (已更新) ---
# 修改为您的 CSV 文件路径
DATA_FILE_PATH = "/home/users/nus/e1352533/mydata/trans_dic/stardict.csv"  # <--- 请将这里替换为您的 CSV 文件实际路径
TOKENIZER_PATH = "/home/users/nus/e1352533/code/pretrain-LLM-from-scratch/tokenizer"
# 建议为不同数据集使用不同输出目录
PROCESSED_DATA_PATH = "/home/users/nus/e1352533/mydata/trans_dic/stardict_processed_data"
BLOCK_SIZE = 4096
NUM_PROC = os.cpu_count() // 2 or 8

def format_dictionary_entry(example):
    """
    将一个词典条目（一行数据）格式化为一段有意义的文本。
    这样有助于模型学习单词、发音、词性、释义和翻译之间的关系。
    """
    parts = []
    # 确保核心字段存在且不为空
    word = example.get("word")
    if word and isinstance(word, str):
        parts.append(f"词：{word}")

        phonetic = example.get("phonetic")
        if phonetic and isinstance(phonetic, str):
            parts.append(f"发音：[{phonetic}]")
        
        pos = example.get("pos")
        if pos and isinstance(pos, str):
            parts.append(f"词性：{pos}")

        definition = example.get("definition")
        if definition and isinstance(definition, str):
            # definition 字段可能包含换行符，我们将其替换为空格以保持格式整洁
            clean_definition = definition.replace('\n', ' ').strip()
            parts.append(f"释义：{clean_definition}")

        translation = example.get("translation")
        if translation and isinstance(translation, str):
            # translation 字段也可能包含换行符
            clean_translation = translation.replace('\n', ' ').strip()
            parts.append(f"翻译：{clean_translation}")

    # 使用换行符将各个部分连接成一个完整的条目文本
    return "\n".join(parts)


def prepare_and_tokenize_dataset():
    """
    加载、格式化、编码并块化词典数据集（仅保存 input_ids）。
    """

    print("=== 步骤 1/5: 定位 CSV 数据文件 ===")
    if not os.path.exists(DATA_FILE_PATH):
        raise ValueError(f"在路径 {DATA_FILE_PATH} 中没有找到 CSV 文件")
    print(f"找到数据文件: {DATA_FILE_PATH}")

    print("\n=== 步骤 2/5: 加载 CSV 数据集 ===")
    # 使用 "csv" 加载器，不需要指定 features，会自动推断
    raw_ds = load_dataset(
        "csv",
        data_files=DATA_FILE_PATH,
        split="train",
        streaming=False, # 保持 False以便多进程处理和获取长度
    )
    
    # 确保数据非空
    if len(raw_ds) == 0:
        raise ValueError("数据集为空，请检查 CSV 文件内容")
        
    print(f"原始数据集加载完成，共 {len(raw_ds)} 条记录。")
    print(f"列名: {raw_ds.column_names}")
    print(f"示例行: {raw_ds[0]}")


    print("\n=== 步骤 3/5: 筛选有效样本 ===")
    # 筛选标准：必须包含 'word'，并且 'definition' 或 'translation' 至少有一个不为空
    before = len(raw_ds)
    filtered_ds = raw_ds.filter(
        lambda x: (x["word"] and isinstance(x["word"], str)) and \
                  ((x["definition"] and isinstance(x["definition"], str)) or \
                   (x["translation"] and isinstance(x["translation"], str))),
        num_proc=NUM_PROC,
        desc="Filtering for valid entries (word + definition/translation)",
    )
    after = len(filtered_ds)
    if after == 0:
        raise ValueError("筛选后无有效样本。请检查数据质量或筛选条件。")
    print(f"筛选完成：{before} -> {after} 条")

    print("\n=== 步骤 4/5: 加载 Tokenizer 并进行分词 ===")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer 无 pad_token，将其设置为 eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    SEP_TEXT = "</s>" # 使用结束符分隔不同的词典条目

    def tokenize_function(examples):
        # 批量处理：为每行数据生成格式化的文本
        # 注意: map 在 batched=True 模式下，examples 是一个字典，key 是列名，value 是该列的 list
        # 我们需要重新组合它们
        formatted_texts = []
        # 获取批次大小
        batch_size = len(examples[list(examples.keys())[0]])
        for i in range(batch_size):
            # 为当前批次中的第 i 个样本构建一个字典
            entry_dict = {key: examples[key][i] for key in examples.keys()}
            # 格式化该样本并添加分隔符
            formatted_text = format_dictionary_entry(entry_dict) + SEP_TEXT
            formatted_texts.append(formatted_text)
            
        # 对格式化后的文本进行分词
        encoded = tokenizer(
            formatted_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return {"input_ids": encoded["input_ids"]}

    tokenized_ds = filtered_ds.map(
        tokenize_function,
        batched=True,
        batch_size=1000, # 可以根据内存调整
        num_proc=NUM_PROC,
        remove_columns=filtered_ds.column_names,
        desc="Formatting and Tokenizing",
    )
    print(f"分词完成：样本数 = {len(tokenized_ds)}；列 = {tokenized_ds.column_names}")

    print("\n=== 步骤 5/5: 块化为固定长度（仅保留 input_ids） ===")
    # 这部分逻辑与原代码完全相同，无需修改
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
        batch_size=2000, # 可以根据内存调整
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