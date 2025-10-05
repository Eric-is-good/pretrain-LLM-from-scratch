import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from typing import List, Dict, Tuple, Any

# --- 配置部分 ---
# 输入和输出文件夹
INPUT_DIR = "/home/users/nus/e1520276/scratch/data/mini"  # 替换为你的输入文件夹路径
OUTPUT_DIR = "/home/users/nus/e1520276/scratch/data/mini/aug" # 替换为你的输出文件夹路径

# 模型和分词器配置
MODEL_NAME = "/local_home/zhupeiyin/model/Qwen3-4B/"
TOKENIZER_NAME = MODEL_NAME
TENSOR_PARALLEL_SIZE = 4

# Token 长度限制配置
MAX_INPUT_TOKENS = 2048 + 400
MAX_OUTPUT_TOKENS = 2048
MAX_TOTAL_LEN = MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS + 100

# vLLM采样参数
# 用于单次生成的采样参数
SAMPLING_PARAMS = SamplingParams(
    n=1,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=MAX_OUTPUT_TOKENS
)
# 【新】用于生成两次追问的采样参数
SAMPLING_PARAMS_DOUBLE = SamplingParams(
    n=2, # 生成两个独立的输出
    temperature=0.7, # 可以稍微提高温度以增加多样性
    top_p=0.95,
    top_k=20,
    max_tokens=MAX_OUTPUT_TOKENS
)


# 输出文件配置
MAX_LINES_PER_FILE = 10000

def prepare_and_truncate_prompt(prompt_text: str, tokenizer: AutoTokenizer) -> Tuple[str, bool]:
    """
    准备vLLM接受的输入格式，并对超长输入进行截断。
    返回格式化后的prompt和是否被截断的布尔值。
    """
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    
    was_truncated = False
    if len(input_ids) > MAX_INPUT_TOKENS:
        was_truncated = True
        truncated_ids = input_ids[:MAX_INPUT_TOKENS]
        formatted_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        
    return formatted_prompt, was_truncated


def parse_llm_json_output(output_text: str) -> dict:
    """从LLM的输出中解析JSON对象"""
    try:
        start_index = output_text.find('{')
        end_index = output_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = output_text[start_index : end_index + 1]
            return json.loads(json_str)
        else:
            return None
    except json.JSONDecodeError:
        return None

def load_and_prepare_all_data(tokenizer: AutoTokenizer) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    【V2 升级】
    阶段一：读取所有jsonl文件，准备好所有step 1的prompts和上下文。
    - 能够处理包含可选'system'指令的对话格式。
    - 根据是否存在'system'指令，生成不同的prompt和上下文。
    """
    print("--- 阶段一: 开始读取所有文件并准备初始 Prompts (已升级) ---")
    
    all_prompts = []
    all_contexts = []
    
    total_lines_read = 0
    skipped_format_error = 0
    truncated_count = 0

    jsonl_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.jsonl') and not f.startswith('.')])
    if not jsonl_files:
        print(f"警告: 在目录 {INPUT_DIR} 中没有找到 .jsonl 文件。")
        return [], []

    for filename in tqdm(jsonl_files, desc="读取文件"):
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    total_lines_read += 1
                    try:
                        data = json.loads(line)
                        if data.get("id") and isinstance(data.get("conversations"), list):
                            convs = data["conversations"]
                            offset = 1 if len(convs) > 0 and convs[0].get('from') == 'system' else 0
                            has_system = (offset == 1)

                            if len(convs) >= offset + 2 and convs[offset].get('from') == 'human' and convs[offset + 1].get('from') == 'gpt':
                                data_id = data['id']
                                question_en = convs[offset]['value']
                                answer_en = convs[offset + 1]['value']
                                
                                prompt_text = ""
                                # 【逻辑变更点】根据是否存在 system prompt 生成不同的翻译任务
                                if has_system:
                                    # 只翻译问题
                                    prompt_text = f"""
你是一名专业的翻译专家，擅长将英文内容精准且自然地翻译成简体中文。
你的任务是翻译下面的问题。请严格按照以下JSON格式返回结果，不要添加任何额外的解释或文本：
{{
  "translated_question": "翻译后的问题"
}}

---
英文问题:
{question_en}
---

JSON输出:
"""
                                else:
                                    # 翻译问答对
                                    prompt_text = f"""
你是一名专业的翻译专家，擅长将英文内容精准且自然地翻译成简体中文，并进行适当的文化本地化。
你的任务是翻译下面的问答对。请严格按照以下JSON格式返回结果，不要添加任何额外的解释或文本：
{{
  "translated_question": "翻译后的问题",
  "translated_answer": "翻译后的答案"
}}

---
英文问题:
{question_en}

英文答案:
{answer_en}
---

JSON输出:
"""
                                final_prompt, was_truncated = prepare_and_truncate_prompt(prompt_text, tokenizer)
                                if was_truncated:
                                    truncated_count += 1

                                all_prompts.append(final_prompt)
                                # 【关键】在上下文中传递 has_system 标记
                                all_contexts.append({'data_id': data_id, 'has_system': has_system, 'answer_en': answer_en})
                            else:
                                skipped_format_error += 1
                        else:
                            skipped_format_error += 1
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        skipped_format_error += 1
        except Exception as e:
            print(f"\n错误: 无法处理文件 {filepath}。错误信息: {e}")

    print("\n--- 数据读取与准备阶段统计报告 (已升级) ---")
    print(f"总共扫描行数: {total_lines_read}")
    print(f"成功解析并生成Prompt的数量: {len(all_prompts)}")
    print(f"因格式错误或数据不完整而跳过的行数: {skipped_format_error}")
    print(f"因输入超长而被截断的Prompt数量: {truncated_count}")
    print("-------------------------------------\n")
    
    return all_prompts, all_contexts


def generate_all_steps(llm: LLM, tokenizer: AutoTokenizer, initial_prompts: List[str], initial_contexts: List[Dict]) -> List[Dict]:
    """
    【V2 升级】
    阶段二：接收所有准备好的prompts，分步完成所有LLM生成任务。
    - 根据'has_system'标记执行不同的数据保存和增强策略。
    """
    print("--- 阶段二: 开始执行LLM生成任务 (已升级) ---")
    if not initial_prompts:
        return []

    final_results = []
    json_parse_errors = 0
    
    # --- 步骤 1: 翻译 ---
    print(f"正在执行步骤1: 翻译... (共 {len(initial_prompts)} 条)")
    step1_outputs = llm.generate(initial_prompts, SAMPLING_PARAMS)
    
    step2_prompts = []
    step2_contexts = []
    
    desc_step1 = "处理步骤1结果 (翻译)"
    for output, context in tqdm(zip(step1_outputs, initial_contexts), total=len(initial_prompts), desc=desc_step1):
        parsed_json = parse_llm_json_output(output.outputs[0].text)
        
        q1_cn, a1_cn = None, None
        
        if parsed_json and "translated_question" in parsed_json:
            q1_cn = parsed_json["translated_question"]
            
            # 【逻辑变更点】根据 has_system 决定如何处理
            if context['has_system']:
                # 如果有system, 我们只需要翻译后的问题q1_cn继续流程
                # a1_cn 保持为 None，并且不保存此翻译对
                pass 
            elif "translated_answer" in parsed_json:
                # 如果没有system, 正常获取翻译的答案，并保存此问答对
                a1_cn = parsed_json["translated_answer"]
                final_results.append({"instruction": q1_cn, "output": a1_cn, "source_id": context['data_id'], "step": 1})
            else:
                json_parse_errors += 1
                continue # 如果没有system但缺少答案，跳过

            # 准备步骤2的输入 (对所有情况都适用)
            prompt_text_step2 = f"""
你是一位知识渊博的AI助手。请根据以下问题，提供一个全面、准确且有条理的中文回答。

请严格按照以下JSON格式返回结果，不要添加任何额外的解释或文本：
{{
  "regenerated_answer": "你生成的完整回答"
}}

---
问题: {q1_cn}
---

JSON输出:
"""
            final_prompt_s2, _ = prepare_and_truncate_prompt(prompt_text_step2, tokenizer)
            step2_prompts.append(final_prompt_s2)
            step2_contexts.append({'q1_cn': q1_cn, 'data_id': context['data_id'], 'has_system': context['has_system']})
        else:
            json_parse_errors += 1

    # --- 步骤 2: 重新生成答案 ---
    if not step2_prompts:
        print("步骤1没有成功生成任何结果，无法进行后续步骤。")
        return final_results
        
    print(f"\n正在执行步骤2: 重新生成答案... (共 {len(step2_prompts)} 条)")
    step2_outputs = llm.generate(step2_prompts, SAMPLING_PARAMS)

    # 【逻辑变更点】为步骤3分离不同采样策略的输入
    step3_prompts_normal, step3_contexts_normal = [], []
    step3_prompts_double, step3_contexts_double = [], []

    desc_step2 = "处理步骤2结果 (重写答案)"
    for output, context in tqdm(zip(step2_outputs, step2_contexts), total=len(step2_prompts), desc=desc_step2):
        parsed_json = parse_llm_json_output(output.outputs[0].text)
        if parsed_json and "regenerated_answer" in parsed_json:
            a2_cn = parsed_json["regenerated_answer"].strip()
            
            # 保存步骤2的结果 (对所有情况都适用)
            final_results.append({"instruction": context['q1_cn'], "output": a2_cn, "source_id": context['data_id'], "step": 2})

            # 【逻辑变更点】准备步骤3的输入，只依赖问题
            prompt_text_step3 = f"""
你是一位富有创造力和逻辑性的AI助手，擅长在对话中进行延伸和拓展。
基于下面的问题，请你提出一个与主题紧密相关、有深度或扩展性的新问题，并对这个新问题进行详细解答。

请严格按照以下JSON格式返回结果，不要添加任何额外的解释或文本：
{{
  "follow_up_question": "你提出的新问题",
  "follow_up_answer": "你对新问题的解答"
}}

---
问题:
{context['q1_cn']}
---

JSON输出:
"""
            final_prompt_s3, _ = prepare_and_truncate_prompt(prompt_text_step3, tokenizer)
            
            # 根据 has_system 分流
            if context['has_system']:
                step3_prompts_double.append(final_prompt_s3)
                step3_contexts_double.append({'data_id': context['data_id']})
            else:
                step3_prompts_normal.append(final_prompt_s3)
                step3_contexts_normal.append({'data_id': context['data_id']})
        else:
            json_parse_errors += 1

    # --- 步骤 3: 创建追问 ---
    # Part A: 处理普通追问 (n=1)
    if step3_prompts_normal:
        print(f"\n正在执行步骤3 (Part A): 创建单次追问... (共 {len(step3_prompts_normal)} 条)")
        step3_outputs_normal = llm.generate(step3_prompts_normal, SAMPLING_PARAMS)
        desc_s3a = "处理步骤3结果 (单次追问)"
        for output, context in tqdm(zip(step3_outputs_normal, step3_contexts_normal), total=len(step3_outputs_normal), desc=desc_s3a):
            parsed_json = parse_llm_json_output(output.outputs[0].text)
            if parsed_json and "follow_up_question" in parsed_json and "follow_up_answer" in parsed_json:
                final_results.append({
                    "instruction": parsed_json["follow_up_question"], 
                    "output": parsed_json["follow_up_answer"], 
                    "source_id": context['data_id'], 
                    "step": 3
                })
            else:
                json_parse_errors += 1
    
    # Part B: 处理需要数据平衡的追问 (n=2)
    if step3_prompts_double:
        print(f"\n正在执行步骤3 (Part B): 创建两次追问以平衡数据... (共 {len(step3_prompts_double)} 条)")
        step3_outputs_double = llm.generate(step3_prompts_double, SAMPLING_PARAMS_DOUBLE)
        desc_s3b = "处理步骤3结果 (两次追问)"
        for output, context in tqdm(zip(step3_outputs_double, step3_contexts_double), total=len(step3_outputs_double), desc=desc_s3b):
            # 【关键】一个输入对应多个输出 (n=2)
            for single_output in output.outputs:
                parsed_json = parse_llm_json_output(single_output.text)
                if parsed_json and "follow_up_question" in parsed_json and "follow_up_answer" in parsed_json:
                    final_results.append({
                        "instruction": parsed_json["follow_up_question"], 
                        "output": parsed_json["follow_up_answer"], 
                        "source_id": context['data_id'], 
                        "step": 3
                    })
                else:
                    json_parse_errors += 1

    print("\n--- LLM 生成阶段统计报告 (已升级) ---")
    print(f"所有步骤累计JSON解析失败或格式不符的数量: {json_parse_errors}")
    print("-------------------------------------\n")

    return final_results


def save_all_results(results: list):
    """
    阶段三：将所有生成的结果分块写入文件。
    """
    print("--- 阶段三: 开始保存所有结果 ---")
    if not results:
        print("没有结果需要保存。")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    total_saved = 0
    num_files = (len(results) + MAX_LINES_PER_FILE - 1) // MAX_LINES_PER_FILE
    
    for i in range(num_files):
        start_index = i * MAX_LINES_PER_FILE
        end_index = start_index + MAX_LINES_PER_FILE
        chunk = results[start_index:end_index]
        
        output_filename = os.path.join(OUTPUT_DIR, f"localized_data_{i + 1}.jsonl")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                for item in chunk:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"成功保存 {len(chunk)} 条数据到文件: {output_filename}")
            total_saved += len(chunk)
        except Exception as e:
            print(f"错误: 写入文件 {output_filename} 失败。错误信息: {e}")

    print("\n--- 保存阶段总结 ---")
    print(f"总共成功保存了 {total_saved} 条增强后的数据。")
    print("---------------------\n")


def main():
    """主执行函数"""
    print(f"正在加载模型: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        max_model_len=MAX_TOTAL_LEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print("模型加载完成。")

    initial_prompts, initial_contexts = load_and_prepare_all_data(tokenizer)
    final_results = generate_all_steps(llm, tokenizer, initial_prompts, initial_contexts)
    save_all_results(final_results)
    
    print("所有任务完成！")


if __name__ == "__main__":
    main()
