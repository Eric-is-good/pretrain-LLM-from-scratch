# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
from datasets import Dataset
import json
import re
import os

##########################################################################################################
# process data
FORMART_PROMPT = """
请按照这样的格式回答问题:
<think>
详细思考过程
</think>
<answer>
简短的先总结，再回答，最终答案是一个数字，使用'#### 数字'表示 
</answer>
"""

# Load the JSON file
file_path = 'data.jsonl'  # Replace with the path to your uploaded JSON file

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    return data

data = load_jsonl(file_path)

data_dict = {
    ('prompt' if key == 'question' else key): 
    [
        [{"role": "user", "content": FORMART_PROMPT + item[key]}] if key == 'question' else
        [{"role": "assistant", "content": item[key]}] for item in data
    ]
    for key in data[0].keys()
}

dataset = Dataset.from_dict(data_dict)

print(dataset)
print(dataset[0])


# ##########################################################################################################
# # reward model/fuctions ( only fuctions @_@ )

def length_reward_func(completions, **kwargs):
    score = [float(len(completion[0]["content"]))*0.002 for completion in completions]
    
    # print("##############################")
    # print(score)
    
    return score

def format_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    score = [1.5 if match else -0.5 for match in matches]
    
    # print("##############################")
    # print(score)
    
    return score

def answer_reward_func(completions, answer, **kwargs):
    completions_matches = [re.search(r"####\s*(\d+)", str(c)) for c in completions]
    completions_ = [match.group(1) if match else "" for match in completions_matches]
    
    ground_truth_matches = [re.search(r"####\s*(\d+)", str(c)) for c in answer]
    ground_truth_ = [match.group(1) if match else "" for match in ground_truth_matches]

    score = [3.0 if c.strip() == gt.strip() else -1.0 for c, gt in zip(completions_, ground_truth_)]
    
    # print("##############################")
    # print(score)
    
    # 选择一个得了3分且 completions 最长的打印
    # Find completions with a score of 3
    best_answer = None
    max_length = 0
    
    for c, s in zip(completions, score):
        if s == 3.0:  # If the score is 3
            if len(c) > max_length:  # Check if it's the longest one
                max_length = len(c)
                best_answer = c

    # Print the best match (longest one that scored 3)
    if best_answer:
        print(best_answer)
    
    
    return score

##########################################################################################################
# training args
os.environ["WANDB_PROJECT"] = "TRL"

training_args = GRPOConfig(
    output_dir="GRPO", 
    run_name="GRPO",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    use_vllm=True,
    max_prompt_length=1024,
    max_completion_length=2048,
    bf16=True,
    save_steps=1000,
    max_grad_norm=1.0,
    report_to="wandb",
    logging_steps=1
)

trainer = GRPOTrainer(
    model="model/",
    reward_funcs=[length_reward_func, format_reward_func, answer_reward_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
