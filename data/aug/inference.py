# conda activate /local_home/zhupeiyin/envs/vllm

import os
import json
import random
import glob
import inspect
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from prompt_templates import WikiPromptTemplates

# --- 1. 配置区域 (已更新) ---

# 模型和路径配置
MODEL_NAME = "/local_home/zhupeiyin/model/Qwen3-4B"
INPUT_DATA_DIR = "/local_home/zhupeiyin/data/wiki/"
OUTPUT_DATA_DIR = "/local_home/zhupeiyin/data/wiki/augmented_data"

# 推理与批处理配置
BATCH_SIZE = 1024*4
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.0 # 稍微降低惩罚，以允许更多样化的表达
MAX_TOKENS = 2048
CUT_OFF_LEN = 4096
MAX_TOTAL_LEN = MAX_TOKENS + CUT_OFF_LEN

# --- 新增：数据增强策略配置 ---
NUM_AUGMENTATIONS_PER_SNIPPET = 3  # 为每个文本片段生成3种不同的增强数据
MIN_TOKENS_FOR_SUMMARIZE = 200     # 仅当文本Token数超过200时才使用摘要模板

# --- 2. 初始化模型和分词器 ---

print(f"正在初始化模型: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(model=MODEL_NAME, trust_remote_code=True, tensor_parallel_size=4, max_model_len=MAX_TOTAL_LEN)
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    max_tokens=MAX_TOKENS,
    # presence_penalty=PRESENCE_PENALTY,
)
print("模型初始化完成。")


# --- 3. 核心功能函数 (已重构和新增) ---

def get_all_template_functions():
    """
    动态获取WikiPromptTemplates类中所有模板，并以字典形式返回。
    返回格式: {'template_name': <function_object>}
    """
    templates = {
        name: func
        for name, func in inspect.getmembers(WikiPromptTemplates, inspect.isfunction)
    }
    if not templates:
        raise ValueError("在 'prompt_templates.py' 中没有找到任何模板函数!")
    print(f"成功加载 {len(templates)} 个Prompt模板: {list(templates.keys())}")
    return templates

def select_templates_for_snippet(all_templates, token_count):
    """
    根据策略为给定的文本片段智能选择模板。
    - Args:
    -   all_templates (dict): 所有可用模板的字典。
    -   token_count (int): 当前文本片段的Token数量。
    - Returns:
    -   list: 包含选定模板函数对象的列表。
    """
    candidate_templates = list(all_templates.values())

    # --- 条件化规则应用 ---
    # 规则1: 如果token数量小于阈值，则排除摘要任务
    if token_count < MIN_TOKENS_FOR_SUMMARIZE:
        summarize_func = all_templates.get('summarize')
        if summarize_func and summarize_func in candidate_templates:
            candidate_templates.remove(summarize_func)
            # print(f"DEBUG: 文本长度 {token_count} < {MIN_TOKENS_FOR_SUMMARIZE}, 已排除 'summarize' 模板。") # 可选的调试信息

    # 确保请求的模板数量不超过候选模板的总数
    num_to_select = min(NUM_AUGMENTATIONS_PER_SNIPPET, len(candidate_templates))

    if num_to_select == 0:
        return []

    # 从候选列表中随机选择不重复的N个模板
    return random.sample(candidate_templates, k=num_to_select)


def process_and_save_batch(batch_prompts, batch_metadata, output_f):
    """使用vLLM处理一个批次的prompts并保存结果 (此函数保持不变)"""
    if not batch_prompts:
        return 0

    outputs = llm.generate(batch_prompts, sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        metadata = batch_metadata[i]

        result_record = {
            "original_text_snippet": metadata["original_text"][:500] + "...",
            "augmentation_task": metadata["task_name"],
            "prompt_used": output.prompt,
            "generated_data": generated_text
        }
        output_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    return len(outputs)


# --- 4. 主执行逻辑 (已更新) ---

def main():
    """主执行函数"""
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    # 获取所有模板，并以名称->函数的字典形式存储
    all_template_functions = get_all_template_functions()

    input_files = glob.glob(os.path.join(INPUT_DATA_DIR, "*.jsonl"))
    if not input_files:
        print(f"错误: 在文件夹 '{INPUT_DATA_DIR}' 中没有找到任何 .jsonl 文件。")
        return

    print(f"找到 {len(input_files)} 个输入文件，将为每个文本片段生成 {NUM_AUGMENTATIONS_PER_SNIPPET} 条增强数据。")

    total_processed_count = 0
    for input_file_path in input_files:
        file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(OUTPUT_DATA_DIR, f"augmented_{file_name}")

        print(f"\n正在处理文件: {file_name}")

        batch_prompts_for_llm = []
        batch_metadata = []

        with open(input_file_path, 'r', encoding='utf-8') as in_f, \
             open(output_file_path, 'w', encoding='utf-8') as out_f:

            for line in tqdm(in_f, desc=f"处理 {file_name}"):
                try:
                    data = json.loads(line)
                    wiki_text = data.get("text")
                    if not wiki_text:
                        continue

                    # 1. 先进行分词以获取真实的token数量，用于策略判断
                    tokenized_output = tokenizer(
                        wiki_text,
                        truncation=False, # 此处先不截断，以获得原始长度
                        return_tensors="pt"
                    )
                    token_ids = tokenized_output.input_ids[0]
                    token_count = len(token_ids)

                    # 2. 根据策略选择要应用的模板
                    selected_templates = select_templates_for_snippet(all_template_functions, token_count)
                    if not selected_templates:
                        continue

                    # 3. 准备用于模板的文本 (截断)
                    # 保留500个token的余量给prompt模板本身
                    if token_count > CUT_OFF_LEN - 500:
                         token_ids = token_ids[:CUT_OFF_LEN - 500]

                    text_for_prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

                    # 4. 为每个选中的模板生成一个prompt，并加入批处理队列
                    for template_func in selected_templates:
                        prompt_content = template_func(text_for_prompt)

                        messages = [{"role": "user", "content": prompt_content}]
                        final_prompt_str = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )

                        batch_prompts_for_llm.append(final_prompt_str)
                        batch_metadata.append({
                            "original_text": text_for_prompt,
                            "task_name": template_func.__name__
                        })

                        # 如果达到批处理大小，则处理并清空批次
                        if len(batch_prompts_for_llm) >= BATCH_SIZE:
                            count = process_and_save_batch(batch_prompts_for_llm, batch_metadata, out_f)
                            total_processed_count += count
                            batch_prompts_for_llm.clear()
                            batch_metadata.clear()

                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行: {line.strip()}")
                    continue

            # 处理文件末尾剩余的最后一个批次
            if batch_prompts_for_llm:
                count = process_and_save_batch(batch_prompts_for_llm, batch_metadata, out_f)
                total_processed_count += count

        print(f"文件 {file_name} 处理完成，结果已保存至 {output_file_path}")

    print(f"\n所有任务完成！总共生成了 {total_processed_count} 条增强数据。")


if __name__ == "__main__":
    main()