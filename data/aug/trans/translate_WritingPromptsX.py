import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import re

# --- 配置参数 ---
INPUT_PARQUET_PATH = "/home/users/nus/e1352533/scratch/data/trans/WritingPromptsX.parquet" 
MODEL_NAME = "/home/users/nus/e1352533/scratch/model/Qwen3-4B" 
OUTPUT_FILE = "/home/users/nus/e1352533/scratch/data/trans/WritingPromptsX_zh.parquet"
TENSOR_PARALLEL_SIZE = 2 
MAX_TOTAL_LEN = 4096 * 2
MAX_NEW_TOKENS = 4096 
MAX_INPUT_TOKENS = 4096 
NUM_SAMPLES = None
# --- 新增参数: Body字段的最小Token数要求 ---
MIN_BODY_TOKENS = 100 

# --- Prompt (保持不变) ---
SYSTEM_PROMPT = """你是一位经验丰富的文学翻译家，擅长将英文创意写作任务翻译成中文。
你的翻译风格追求“信、达、雅”，力求在忠于原文的基础上，使其语言优美、富有文采、充满韵味，并且完全符合中文的表达习惯，避免任何机器翻译的生硬感。
请严格按照以下格式输出翻译结果，不要包含任何额外的解释或说明：

【标题翻译】
[这里是翻译后的标题]

【正文翻译】
[这里是翻译后的正文]
"""

# --- 主执行流程 ---

def main():
    print(f"正在初始化模型: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def create_combined_translation_prompt(title: str, body: str) -> str:
        content_to_translate = f"【原始标题】\n{title}\n\n【原始正文】\n{body}"
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + f"\n\n请将以下写作任务翻译成中文：\n\n---\n\n{content_to_translate}\n\n---"}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        max_model_len=MAX_TOTAL_LEN,
    )
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=MAX_NEW_TOKENS
    )

    print(f"正在从本地文件加载数据: {INPUT_PARQUET_PATH}...")
    try:
        df_original = pd.read_parquet(INPUT_PARQUET_PATH)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{INPUT_PARQUET_PATH}'。请检查路径是否正确。")
        return

    if NUM_SAMPLES and NUM_SAMPLES < len(df_original):
        df_to_process = df_original.head(NUM_SAMPLES)
        print(f"将处理前 {NUM_SAMPLES} 条样本。")
    else:
        df_to_process = df_original
        print(f"将处理全部 {len(df_original)} 条数据。")

    prompts_to_generate = []
    valid_data_to_process = [] 
    long_text_count = 0
    short_text_count = 0 # 新增：用于统计过短文本的数量

    print("正在构造合并翻译 prompts 并筛选数据...")
    for row in tqdm(df_to_process.itertuples(), total=len(df_to_process), desc="准备数据"):
        post_title = row.post_title
        body = row.body
        
        # 检查是否为有效字符串
        if not (isinstance(post_title, str) and isinstance(body, str) and post_title and body):
            continue

        body_token_ids = tokenizer.encode(body)
        
        # --- 核心修改点: 增加长度筛选 ---
        if len(body_token_ids) <= MIN_BODY_TOKENS:
            short_text_count += 1
            continue # 如果不满足最小长度要求，则直接跳过该行，不进行任何处理

        # 处理过长文本（截断）
        if len(body_token_ids) > MAX_INPUT_TOKENS:
            long_text_count += 1
            truncated_ids = body_token_ids[:MAX_INPUT_TOKENS]
            body = tokenizer.decode(truncated_ids, skip_special_tokens=True)

        # 如果数据有效且长度符合要求，则创建prompt并保存原文
        combined_prompt = create_combined_translation_prompt(post_title, body)
        prompts_to_generate.append(combined_prompt)
        valid_data_to_process.append({'post_title': post_title, 'body': body})

    # 打印筛选和截断的统计信息
    if short_text_count > 0:
        print(f"提示：根据筛选条件 (正文 > {MIN_BODY_TOKENS} tokens)，已跳过 {short_text_count} 条过短的文本。")
    if long_text_count > 0:
        print(f"注意：检测到并截断了 {long_text_count} 条过长的文本。")
    
    if not prompts_to_generate:
        print("经过筛选后，没有找到有效的可翻译数据，程序退出。")
        return

    print(f"\n开始批量翻译 {len(prompts_to_generate)} 条数据...")
    outputs = llm.generate(prompts_to_generate, sampling_params)
    print("翻译完成。")

    # --- 整理并保存结果 (逻辑不变) ---
    translated_results = []
    print("正在整理结果...")
    for i, output in enumerate(tqdm(outputs, desc="解析结果")):
        generated_text = output.outputs[0].text.strip()
        
        cleaned_text = generated_text.replace("【标题翻译】", "").replace("【正文翻译】", "\n").strip()
        
        original = valid_data_to_process[i]
        translated_results.append({
            'post_title': original['post_title'],
            'body': original['body'],
            'zh': cleaned_text
        })

    print(f"\n正在将结果保存到: {OUTPUT_FILE}...")
    df_final = pd.DataFrame(translated_results)
    df_final.to_parquet(OUTPUT_FILE, index=False)
    
    print("所有任务完成！")
    print("\n预训练语料预览 (最终列为 'post_title', 'body', 'zh'):")
    print(df_final.head())


if __name__ == "__main__":
    main()