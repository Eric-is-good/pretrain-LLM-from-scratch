import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import os

# --- 配置参数 ---
### <<< 修改注释，明确输入为中文，输出为英文
# 输入文件路径，包含待翻译的中文 'content' 列
INPUT_PARQUET_PATH = "/home/users/nus/e1352533/scratch/data/trans/zh-003.parquet" 
# 您本地的 Qwen 模型路径 (Qwen模型是中英双语的，无需更换)
MODEL_NAME = "/home/users/nus/e1352533/scratch/model/Qwen3-4B" 
# 输出文件路径，将保存英文翻译结果
OUTPUT_FILE = "/home/users/nus/e1352533/scratch/data/trans/zh-003-trans-en.parquet"
# 要翻译的目标样本数量（在满足长度条件后）
MAX_SAMPLES_TO_TRANSLATE = 80000 
# 内容字段的最小Token数要求 (对中文同样适用)
MIN_CONTENT_TOKENS = 0 

# --- VLLM & 模型配置 ---
TENSOR_PARALLEL_SIZE = 2 
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 4096 
MAX_INPUT_TOKENS = 4096 

### <<< 核心修改点：更新为中译英的Prompt ###
# --- Prompt ---
SYSTEM_PROMPT = """You are an expert literary translator, specializing in translating Chinese texts into English.
Your translation should be faithful to the original, while also being fluent, elegant, and idiomatic. 
The goal is to produce a translation that reads naturally to a native English speaker, completely avoiding any stiff or literal "Chinglish" phrasing.
Please provide only the translated English text in your response, without any additional explanations, labels (like "[Translation]"), or introductory phrases.
"""

# --- 主执行流程 ---

def main():
    """主执行函数"""
    print(f"正在初始化模型: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def create_translation_prompt(content: str) -> str:
        """为单段内容创建翻译prompt"""
        # ### <<< 修改: Prompt现在是英文的，所以system角色内容也变成英文
        # ### <<< user角色的内容依然是待翻译的原文（中文）
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + content}
        ]
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

    ### <<< 修改打印信息，明确任务方向
    print(f"正在从本地加载中文数据: {INPUT_PARQUET_PATH}...")
    try:
        df_original = pd.read_parquet(INPUT_PARQUET_PATH)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{INPUT_PARQUET_PATH}'。请检查路径是否正确。")
        return

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
    for row in tqdm(df_original.itertuples(), total=len(df_original), desc="筛选数据"):
        if len(prompts_to_generate) >= MAX_SAMPLES_TO_TRANSLATE:
            print(f"\n已找到 {MAX_SAMPLES_TO_TRANSLATE} 条符合条件的样本，停止筛选。")
            break
            
        content = getattr(row, 'content', None)
        processed_rows_count += 1
        
        if not isinstance(content, str) or not content:
            continue

        token_ids = tokenizer.encode(content)
        
        if len(token_ids) <= MIN_CONTENT_TOKENS:
            short_text_count += 1
            continue

        if len(token_ids) > MAX_INPUT_TOKENS:
            long_text_count += 1
            truncated_ids = token_ids[:MAX_INPUT_TOKENS]
            content = tokenizer.decode(truncated_ids, skip_special_tokens=True)

        prompt = create_translation_prompt(content)
        prompts_to_generate.append(prompt)
        valid_original_content.append(content)

    print("\n--- 数据筛选统计 ---")
    print(f"总共检查了 {processed_rows_count} 行数据。")
    print(f"跳过了 {short_text_count} 条过短的文本 (<= {MIN_CONTENT_TOKENS} tokens)。")
    if long_text_count > 0:
        print(f"截断了 {long_text_count} 条过长的文本 (输入 > {MAX_INPUT_TOKENS} tokens)。")
    print(f"最终准备翻译 {len(prompts_to_generate)} 条有效数据。")
    
    if not prompts_to_generate:
        print("经过筛选后，没有找到有效的可翻译数据，程序退出。")
        return

    ### <<< 修改打印信息
    print(f"\n开始将 {len(prompts_to_generate)} 条中文数据批量翻译为英文...")
    outputs = llm.generate(prompts_to_generate, sampling_params)
    print("翻译完成。")

    translated_results = []
    print("正在整理结果...")
    for i, output in enumerate(tqdm(outputs, desc="解析结果")):
        generated_text = output.outputs[0].text.strip()
        original_content = valid_original_content[i]
        
        translated_results.append({
            'content': original_content, # 中文原文
            'trans': generated_text      # 英文译文
        })

    print(f"\n正在将结果保存到: {OUTPUT_FILE}...")
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df_final = pd.DataFrame(translated_results)
    df_final.to_parquet(OUTPUT_FILE, index=False)
    
    print("所有任务完成！")
    print(f"\n成功保存 {len(df_final)} 行翻译结果。")
    ### <<< 修改预览信息的描述
    print("\n最终文件预览 (列为 'content' [中文原文] 和 'trans' [英文译文]):")
    print(df_final.head())


if __name__ == "__main__":
    main()