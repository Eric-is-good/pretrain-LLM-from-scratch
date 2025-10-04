import os
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain

# --- 1. 配置参数 ---
# 输入文件：指向第一个脚本生成的 Parquet 文件
INPUT_DATA_PATH = "/home/users/nus/e1352533/mydata/data/trans/WritingPromptsX_zh.parquet"
# Tokenizer 路径 (保持不变)
TOKENIZER_PATH = "/home/users/nus/e1352533/code/pretrain-LLM-from-scratch/tokenizer"
# 处理后数据的保存路径
PROCESSED_DATA_PATH = "/home/users/nus/e1352533/mydata/data/trans/WritingPromptsX/"
# 文本块大小 (保持不变)
BLOCK_SIZE = 4096
# 使用的 CPU 核心数
NUM_PROC = os.cpu_count() // 2 or 8

# --- 新增功能参数 ---
# 每条数据被构造成“翻译任务”的概率。
# 剩余的 (1.0 - PROBABILITY) 将被拆分为两个独立的句子进行训练。
TRANSLATION_TASK_PROBABILITY = 0.7


# --- 2. 定义多样化的指令模板 (已扩充) ---

# 英译中 (English to Chinese) 的模板
E2C_TEMPLATES = [
    "Translate the following English text to Chinese.\n\nEnglish: {english}\n\nChinese: {chinese}",
    "将以下英文翻译成中文：\n英文：{english}\n中文：{chinese}",
    "Please provide the Chinese translation for the following English text:\n\n{english}\n\n---\n\n{chinese}",
    "Task: English to Chinese Translation.\nInput: {english}\nOutput: {chinese}",
    "English: {english}\nChinese: {chinese}",
    "英文：{english}\n中文：{chinese}",
    "{english}\n\nTranslate into Chinese:\n\n{chinese}",
    "{english}\n\n翻译成中文：\n\n{chinese}",
    "Below is an English passage. Your task is to render it into elegant Chinese.\n\n{english}\n\n{chinese}",
]

# 中译英 (Chinese to English) 的模板
C2E_TEMPLATES = [
    "Translate the following Chinese text to English.\n\nChinese: {chinese}\n\nEnglish: {english}",
    "将以下中文翻译成英文：\n中文：{chinese}\n英文：{english}",
    "Please provide the English translation for the following Chinese text:\n\n{chinese}\n\n---\n\n{english}",
    "Task: Chinese to English Translation.\nInput: {chinese}\nOutput: {english}",
    "Chinese: {chinese}\nEnglish: {english}",
    "中文：{chinese}\n英文：{english}",
    "{chinese}\n\nTranslate into English:\n\n{english}",
    "{chinese}\n\n翻译成英文：\n\n{english}",
    "Translate the Chinese content into English.\n\nChinese Source:\n{chinese}\n\nEnglish Translation:\n{english}",
]


def prepare_and_tokenize_dataset():
    """
    加载、增强、编码并块化处理 WritingPrompts 数据集。
    """
    print("=== 步骤 1/6: 加载 Parquet 数据集 ===")
    raw_ds = load_dataset("parquet", data_files=INPUT_DATA_PATH, split="train")

    # 确保存在所需列
    required_cols = {"post_title", "body", "zh"}
    if not required_cols.issubset(raw_ds.column_names):
        raise ValueError(f"数据缺失必须列: {required_cols - set(raw_ds.column_names)}")
    print(f"加载完成，共 {len(raw_ds)} 条数据。")

    print("\n=== 步骤 2/6: 筛选无效行 ===")
    before = len(raw_ds)
    # 筛选掉 "post_title", "body", 或 "zh" 为空或非字符串的行
    filtered_ds = raw_ds.filter(
        lambda x: (
            x["post_title"] and isinstance(x["post_title"], str) and
            x["body"] and isinstance(x["body"], str) and
            x["zh"] and isinstance(x["zh"], str)
        ),
        num_proc=NUM_PROC,
        desc="Filtering invalid rows"
    )
    after = len(filtered_ds)
    print(f"筛选完成：{before} -> {after} 条")
    if after == 0:
        raise ValueError("筛选后无有效样本，请检查文件内容。")

    print("\n=== 步骤 3/6: 应用双重任务模式进行数据增强 ===")
    print(f"每行数据将有 {TRANSLATION_TASK_PROBABILITY:.0%} 的概率被构造成翻译任务，")
    print(f"否则将被拆分为两个独立的句子用于语言模型预训练。")

    def format_data_with_dual_tasks(examples):
        processed_texts = []
        # batched=True 时, examples 是一个字典，其值为列表
        for i in range(len(examples["post_title"])):
            # 1. 合并 title 和 body 成为一个整体
            english_text = examples["post_title"][i] + "\n\n" + examples["body"][i]
            chinese_text = examples["zh"][i]

            # 2. 随机决定任务类型
            if random.random() < TRANSLATION_TASK_PROBABILITY:
                # --- 模式 A: 构造成翻译任务 ---
                if random.random() < 0.5:
                    # 英译中
                    template = random.choice(E2C_TEMPLATES)
                    formatted_text = template.format(english=english_text, chinese=chinese_text)
                else:
                    # 中译英
                    template = random.choice(C2E_TEMPLATES)
                    formatted_text = template.format(english=english_text, chinese=chinese_text)
                processed_texts.append(formatted_text)
            else:
                # --- 模式 B: 拆分为两个独立句子 ---
                processed_texts.append(english_text)
                processed_texts.append(chinese_text)

        return {"text": processed_texts}

    formatted_ds = filtered_ds.map(
        format_data_with_dual_tasks,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        remove_columns=["post_title", "body", "zh"], # 移除所有原始列
        desc="Applying dual-task formatting"
    )
    print("数据增强完成。示例:")
    print("---")
    # 打印随机样本查看效果
    sample_indices = random.sample(range(len(formatted_ds)), k=min(3, len(formatted_ds)))
    for i in sample_indices:
        print(f"样本 {i+1}:")
        print(formatted_ds[i]['text'])
        print("---")

    print("\n=== 步骤 4/6: 加载 Tokenizer 并进行分词 ===")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 每个独立的文本样本（无论是翻译任务还是单个句子）结束后都添加结束符
    SEP_TEXT = tokenizer.eos_token

    def tokenize_function(examples):
        texts = [text + SEP_TEXT for text in examples["text"]]
        return tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )

    tokenized_ds = formatted_ds.map(
        tokenize_function,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=formatted_ds.column_names,
        desc="Tokenizing",
    )
    print(f"分词完成。注意：由于双重任务模式，这里的样本数可能多于原始行数。")
    print(f"分词后样本数 = {len(tokenized_ds)}")

    print("\n=== 步骤 5/6: 块化为固定长度 ===")
    def group_texts(examples):
        concatenated_ids = list(chain.from_iterable(examples["input_ids"]))
        total_length = len(concatenated_ids)
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of block_size
        result = [
            concatenated_ids[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)
        ]
        return {"input_ids": result}

    lm_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        num_proc=NUM_PROC,
        desc=f"Grouping into blocks of {BLOCK_SIZE}",
    )
    print(f"块化完成！最终用于训练的样本数（block 数）：{len(lm_ds)}")

    print(f"\n=== 步骤 6/6: 保存到磁盘：{PROCESSED_DATA_PATH} ===")
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    lm_ds.save_to_disk(PROCESSED_DATA_PATH)
    print("✅ 所有数据准备工作已完成。")


if __name__ == "__main__":
    prepare_and_tokenize_dataset()