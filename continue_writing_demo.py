from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

# --- 模型和分词器加载 ---

model = AutoModelForCausalLM.from_pretrained("continue_writing/", trust_remote_code=True).to("cuda") # pip install transformers==4.50.1
tokenizer = AutoTokenizer.from_pretrained("continue_writing/")
model.eval()

# input_text = "秋日午后的公园长椅总藏着温柔的时光。阳光穿过疏朗的梧桐叶，在木质椅面上洒下细碎的金斑，"
input_text = "守护文化遗产是人类对自身文明根脉的敬畏与担当。从故宫红墙下的斗拱飞檐到敦煌莫高窟的飞天壁画，"
# input_text = "算法公平性是人工智能技术健康发展的核心伦理基石，也是避免技术加剧社会偏见的关键防线。"


inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids.to("cuda")

output = model.generate(
    input_ids=input_ids,
    max_length=128,
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