from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# --- 模型和分词器加载 ---

model = AutoModelForCausalLM.from_pretrained("chat/", trust_remote_code=True).to("cuda") # pip install transformers==4.50.1
tokenizer = AutoTokenizer.from_pretrained("chat/")
model.eval()

chat_prompt = """<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
<think>"""

input_text = "请问你是谁？"
# input_text = "守护文化遗产的重要性体现在哪些方面？"
# input_text = "请问什么是算法公平性？"
# input_text = "翻译成中文：Cherry blossom viewing, known as 'hanami' in Japanese, is far more than a casual spring outing."
# input_text = "小明去文具店买学习用品，一支钢笔售价 12 元，一本笔记本售价 5 元。他买了 2 支钢笔和 3 本笔记本，付款时店员给他减免了 3 元。请问小明最终需要支付多少钱？"

input_text = chat_prompt.format(input=input_text)
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids.to("cuda")

# --- 在 generate 函数中应用扩展后的 bad_words_ids ---
output = model.generate(
    input_ids=input_ids,
    max_length=256,
    # repetition_penalty=1.1,
    do_sample=True,
    temperature=1.0,
    top_k=20,
    top_p=0.95,
    num_return_sequences=8
)

print("\n--- 生成结果 ---")
for i, output_seq in enumerate(output):
    print(f"序列 {i + 1}: {tokenizer.decode(output_seq, skip_special_tokens=False)}")
    print("-" * 50)