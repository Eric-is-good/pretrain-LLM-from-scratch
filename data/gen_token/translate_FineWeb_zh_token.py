import os
import random
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain

# --- 1. 配置参数 ---
# 输入文件：【请确保这里指向您新的 Parquet 文件】
INPUT_DATA_PATH = "/home/users/nus/e1352533/mydata/data/trans/zh-003-trans-en.parquet"  # <--- 请务必修改为您的新文件路径

# --- 核心改动：新增文件方向开关 ---
# 根据输入文件的列内容设置方向
# 'en_to_zh': content列是英文, trans列是中文
# 'zh_to_en': content列是中文, trans列是英文
INPUT_FILE_DIRECTION = 'zh_to_en' # <--- 为您的新文件设置为 'zh_to_en'

# Tokenizer 路径 (保持不变)
TOKENIZER_PATH = "/home/users/nus/e1352533/code/pretrain-LLM-from-scratch/tokenizer"
# 处理后数据的保存路径
PROCESSED_DATA_PATH = "/home/users/nus/e1352533/mydata/data/trans/zh-003-trans-en" # 建议为新数据使用新目录
# 文本块大小 (保持不变)
BLOCK_SIZE = 4096
# 使用的 CPU 核心数
NUM_PROC = os.cpu_count() // 2 or 8

# --- 功能参数 ---
TRANSLATION_TASK_PROBABILITY = 1.0
THINK_MODE_PROBABILITY = 0.5

# --- 2. 定义多样化的指令模板 (保持不变) ---
# (模板部分省略，与上一版代码相同)
# 英译中 (English to Chinese) 的模板
E2C_TEMPLATES = [
    "Translate the following English text to Chinese.\n\nEnglish: {english}\n\nChinese: {chinese}",
    "将以下英文翻译成中文：\n英文：{english}\n中文：{chinese}",
    "English: {english}\nChinese: {chinese}", "英文：{english}\n中文：{chinese}",
    "Please provide the Chinese translation for the following English text:\n\n{english}\n\n---\n\n{chinese}",
    "Task: English to Chinese Translation.\nInput: {english}\nOutput: {chinese}",
    "{english}\n\nTranslate into Chinese:\n\n{chinese}",
    "Question: What is the Chinese translation of the following English text?\nEnglish Text: {english}\nAnswer: {chinese}",
]
# 中译英 (Chinese to English) 的模板
C2E_TEMPLATES = [
    "Translate the following Chinese text to English.\n\nChinese: {chinese}\n\nEnglish: {english}",
    "将以下中文翻译成英文：\n中文：{chinese}\n英文：{english}",
    "Chinese: {chinese}\nEnglish: {english}", "中文：{chinese}\n英文：{english}",
    "Please provide the English translation for the following Chinese text:\n\n{chinese}\n\n---\n\n{english}",
    "Task: Chinese to English Translation.\nInput: {chinese}\nOutput: {english}",
    "{chinese}\n\nTranslate into English:\n\n{english}",
    "Question: How would you translate this Chinese text into English?\nChinese: {chinese}\nAnswer: {english}",
]
# 思考模式 (Chain-of-Thought) 模板
E2C_THINK_TEMPLATES = [
    "Translate the following English text to Chinese. Use the think-aloud mode, placing your reasoning process between <think> and </think> tags.\n\nEnglish: {english}\n\nChinese: {chinese}",
    "请使用“思考-输出”模式，将以下英文翻译成中文。请将你的思考过程放在 `<think>` 和 `</think>` 标签之间。\n\n英文：{english}\n\n中文：{chinese}",
    "Task: English to Chinese Translation (with Chain of Thought).\nInstruction: Show your reasoning within <think> tags before the final answer.\nInput: {english}\nOutput: {chinese}",
    "问题：如何将这段英文翻译成中文？请先思考，再给出翻译，并将思考过程写入<think>标签。\n英文：{english}\n回答：{chinese}",
    "Let's think step by step. First, analyze the English sentence, then translate it to Chinese. Wrap your thoughts in <think> tags.\n\n{english}\n\n{chinese}",
]
C2E_THINK_TEMPLATES = [
    "Translate the following Chinese text to English. Use the think-aloud mode, placing your reasoning process between <think> and </think> tags.\n\nChinese: {chinese}\n\nEnglish: {english}",
    "请使用“思考-输出”模式，将以下中文翻译成英文。请将你的思考过程放在 `<think>` 和 `</think>` 标签之间。\n\n中文：{chinese}\n\n英文：{english}",
    "Task: Chinese to English Translation (with Chain of Thought).\nInstruction: Show your reasoning within <think> tags before the final answer.\nInput: {chinese}\nOutput: {english}",
    "问题：如何将这段中文翻译成英文？请先思考，再给出翻译，并将思考过程写入<think>标签。\n中文：{chinese}\n回答：{english}",
]


def strip_think_tags(text):
    """从文本中移除 <think>...</think> 标签及其内容"""
    if isinstance(text, str):
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text

def prepare_and_tokenize_dataset():
    """
    加载、增强、编码并块化处理数据集。
    """
    print("=== 步骤 1/6: 加载 Parquet 数据集 ===")
    raw_ds = load_dataset("parquet", data_files=INPUT_DATA_PATH, split="train")
    required_cols = {"content", "trans"}
    if not required_cols.issubset(raw_ds.column_names):
        raise ValueError(f"数据缺失必须列: {required_cols - set(raw_ds.column_names)}")
    print(f"加载完成，共 {len(raw_ds)} 条数据。")
    print(f"当前文件处理方向: {INPUT_FILE_DIRECTION}")

    print("\n=== 步骤 2/6: 筛选无效行 ===")
    before = len(raw_ds)
    filtered_ds = raw_ds.filter(
        lambda x: (x["content"] and isinstance(x["content"], str) and x["trans"] and isinstance(x["trans"], str)),
        num_proc=NUM_PROC, desc="Filtering invalid rows"
    )
    after = len(filtered_ds)
    print(f"筛选完成：{before} -> {after} 条")
    if after == 0:
        raise ValueError("筛选后无有效样本，请检查文件内容。")

    print("\n=== 步骤 3/6: 应用多模式数据增强 ===")
    print(f"翻译任务概率: {TRANSLATION_TASK_PROBABILITY:.0%}, 思考模式概率: {THINK_MODE_PROBABILITY:.0%}")

    def format_data_with_dual_tasks(examples):
        processed_texts = []
        for i in range(len(examples["content"])):
            
            # --- 核心改动：根据开关分配变量 ---
            if INPUT_FILE_DIRECTION == 'en_to_zh':
                english_text_raw = examples["content"][i]
                chinese_text_raw = examples["trans"][i]
                llm_output_text = chinese_text_raw # trans 列是 LLM 输出
            elif INPUT_FILE_DIRECTION == 'zh_to_en':
                chinese_text_raw = examples["content"][i]
                english_text_raw = examples["trans"][i]
                llm_output_text = english_text_raw # trans 列是 LLM 输出
            else:
                raise ValueError(f"无效的 INPUT_FILE_DIRECTION: {INPUT_FILE_DIRECTION}")

            if random.random() < TRANSLATION_TASK_PROBABILITY:
                # --- 模式 A: 构造成翻译任务 ---
                has_think_tag = "<think>" in llm_output_text and "</think>" in llm_output_text
                use_think_mode = has_think_tag and (random.random() < THINK_MODE_PROBABILITY)
                
                is_e2c = random.random() < 0.5

                if use_think_mode:
                    # 使用思考模式模板，保留原始文本内容
                    template = random.choice(E2C_THINK_TEMPLATES if is_e2c else C2E_THINK_TEMPLATES)
                    formatted_text = template.format(english=english_text_raw, chinese=chinese_text_raw)
                else:
                    # 使用常规模板，需要清理文本中的思考标签
                    template = random.choice(E2C_TEMPLATES if is_e2c else C2E_TEMPLATES)
                    formatted_text = template.format(
                        english=strip_think_tags(english_text_raw), 
                        chinese=strip_think_tags(chinese_text_raw)
                    )
                processed_texts.append(formatted_text)
            else:
                # --- 模式 B: 拆分为两个独立句子 ---
                # 清理掉 think 标签，避免污染预训练语料
                processed_texts.append(strip_think_tags(english_text_raw))
                processed_texts.append(strip_think_tags(chinese_text_raw))

        return {"text": processed_texts}

    formatted_ds = filtered_ds.map(
        format_data_with_dual_tasks,
        batched=True, batch_size=1000, num_proc=NUM_PROC,
        remove_columns=["content", "trans"],
        desc="Applying multi-mode formatting"
    )
    # ... (后续步骤 4, 5, 6 保持不变) ...
    print("数据增强完成。示例:")
    print("---")
    sample_indices = random.sample(range(len(formatted_ds)), k=min(3, len(formatted_ds)))
    for i in sample_indices:
        print(f"样本 {i+1}:")
        print(formatted_ds[i]['text'])
        print("---")

    print("\n=== 步骤 4/6: 加载 Tokenizer 并进行分词 ===")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    SEP_TEXT = tokenizer.eos_token
    def tokenize_function(examples):
        texts = [text + SEP_TEXT for text in examples["text"]]
        return tokenizer(texts, add_special_tokens=False, return_attention_mask=False)

    tokenized_ds = formatted_ds.map(
        tokenize_function, batched=True, num_proc=NUM_PROC,
        remove_columns=formatted_ds.column_names, desc="Tokenizing"
    )
    print(f"分词完成。分词后样本数 = {len(tokenized_ds)}")

    print("\n=== 步骤 5/6: 块化为固定长度 ===")
    def group_texts(examples):
        concatenated_ids = list(chain.from_iterable(examples["input_ids"]))
        total_length = len(concatenated_ids)
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = [concatenated_ids[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        return {"input_ids": result}

    lm_ds = tokenized_ds.map(
        group_texts, batched=True, num_proc=NUM_PROC,
        desc=f"Grouping into blocks of {BLOCK_SIZE}"
    )
    print(f"块化完成！最终用于训练的样本数（block 数）：{len(lm_ds)}")

    print(f"\n=== 步骤 6/6: 保存到磁盘：{PROCESSED_DATA_PATH} ===")
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    lm_ds.save_to_disk(PROCESSED_DATA_PATH)
    print("✅ 所有数据准备工作已完成。")


if __name__ == "__main__":
    prepare_and_tokenize_dataset()