import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import os

# --- 配置参数 ---
# 输入文件路径，包含 'content' 列
INPUT_PARQUET_PATH = "/home/users/nus/e1352533/scratch/data/trans/en-0003.parquet" 
# 您本地的 Qwen 模型路径
MODEL_NAME = "/home/users/nus/e1352533/scratch/model/Qwen3-4B" 
# 输出文件路径
OUTPUT_FILE = "/home/users/nus/e1352533/scratch/data/trans/en-0003-trans-zh.parquet"
# 要翻译的目标样本数量（在满足长度条件后）
MAX_SAMPLES_TO_TRANSLATE = 80000 
# 内容字段的最小Token数要求
MIN_CONTENT_TOKENS = 0

# --- VLLM & 模型配置 ---
TENSOR_PARALLEL_SIZE = 2 
MAX_MODEL_LEN = 8192 # max_model_len = max_input + max_new_tokens
MAX_NEW_TOKENS = 4096 
MAX_INPUT_TOKENS = 4096 

# --- Prompt ---
# 针对单文本翻译进行简化，要求模型直接输出译文，不加任何额外标签
SYSTEM_PROMPT = """你是一位经验丰富的文学翻译家，擅长将英文文本翻译成流畅、地道的中文。
你的翻译风格追求“信、达、雅”，力求在忠于原文的基础上，使其语言优美、富有文采、充满韵味，并且完全符合中文的表达习惯，避免任何机器翻译的生硬感。
请直接开始翻译下面的文本，不要在你的回答中包含任何额外的解释、标签（如“【译文】”）或说明，只输出翻译后的纯中文文本。
"""

# --- 主执行流程 ---

def main():
    """主执行函数"""
    print(f"正在初始化模型: {MODEL_NAME}...")
    # 使用 trust_remote_code 以支持 Qwen 的 Chat Template
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def create_translation_prompt(content: str) -> str:
        """为单段内容创建翻译prompt"""
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + content}
        ]
        # 使用模型的聊天模板来格式化输入
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    print("正在设置 VLLM 引擎...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        max_model_len=MAX_MODEL_LEN,
    )
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=MAX_NEW_TOKENS
    )

    print(f"正在从本地文件加载数据: {INPUT_PARQUET_PATH}...")
    try:
        df_original = pd.read_parquet(INPUT_PARQUET_PATH)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{INPUT_PARQUET_PATH}'。请检查路径是否正确。")
        return

    # 检查 'content' 列是否存在
    if 'content' not in df_original.columns:
        print(f"错误：输入文件中未找到必需的 'content' 列。")
        return
        
    print(f"原始数据集共有 {len(df_original)} 行。")

    prompts_to_generate = []
    valid_original_content = [] 
    long_text_count = 0
    short_text_count = 0
    processed_rows_count = 0

    print(f"正在构造 prompts 并筛选数据 (目标: {MAX_SAMPLES_TO_TRANSLATE} 条, 最小长度: {MIN_CONTENT_TOKENS} tokens)...")
    # 使用 tqdm 显示处理进度
    for row in tqdm(df_original.itertuples(), total=len(df_original), desc="筛选数据"):
        # 已经找到足够数量的样本，提前中断
        if len(prompts_to_generate) >= MAX_SAMPLES_TO_TRANSLATE:
            print(f"\n已找到 {MAX_SAMPLES_TO_TRANSLATE} 条符合条件的样本，停止筛选。")
            break
            
        content = getattr(row, 'content', None)
        processed_rows_count += 1
        
        # 检查是否为有效字符串
        if not isinstance(content, str) or not content:
            continue

        # 检查Token长度
        token_ids = tokenizer.encode(content)
        
        if len(token_ids) <= MIN_CONTENT_TOKENS:
            short_text_count += 1
            continue # 如果不满足最小长度要求，则跳过

        # 处理过长文本（截断）
        if len(token_ids) > MAX_INPUT_TOKENS:
            long_text_count += 1
            truncated_ids = token_ids[:MAX_INPUT_TOKENS]
            content = tokenizer.decode(truncated_ids, skip_special_tokens=True)

        # 如果数据有效且长度符合要求，则创建prompt并保存原文
        prompt = create_translation_prompt(content)
        prompts_to_generate.append(prompt)
        valid_original_content.append(content)

    # 打印筛选和截断的统计信息
    print("\n--- 数据筛选统计 ---")
    print(f"总共检查了 {processed_rows_count} 行数据。")
    print(f"跳过了 {short_text_count} 条过短的文本 (<= {MIN_CONTENT_TOKENS} tokens)。")
    if long_text_count > 0:
        print(f"截断了 {long_text_count} 条过长的文本 (输入 > {MAX_INPUT_TOKENS} tokens)。")
    print(f"最终准备翻译 {len(prompts_to_generate)} 条有效数据。")
    
    if not prompts_to_generate:
        print("经过筛选后，没有找到有效的可翻译数据，程序退出。")
        return

    print(f"\n开始批量翻译 {len(prompts_to_generate)} 条数据...")
    outputs = llm.generate(prompts_to_generate, sampling_params)
    print("翻译完成。")

    # --- 整理并保存结果 ---
    translated_results = []
    print("正在整理结果...")
    for i, output in enumerate(tqdm(outputs, desc="解析结果")):
        generated_text = output.outputs[0].text.strip()
        original_content = valid_original_content[i]
        
        translated_results.append({
            'content': original_content,
            'trans': generated_text
        })

    print(f"\n正在将结果保存到: {OUTPUT_FILE}...")
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df_final = pd.DataFrame(translated_results)
    df_final.to_parquet(OUTPUT_FILE, index=False)
    
    print("所有任务完成！")
    print(f"\n成功保存 {len(df_final)} 行翻译结果。")
    print("\n最终文件预览 (列为 'content' 和 'trans'):")
    print(df_final.head())


if __name__ == "__main__":
    main()