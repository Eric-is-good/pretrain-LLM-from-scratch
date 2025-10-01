from transformers import AutoTokenizer

import sys
sys.path.append("model/")  # 这个需要更改
from modeling_holmes import HolmesForCausalLM

model = HolmesForCausalLM.from_pretrained("final/").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("final/")
model.eval()

input_text = "秋日午后的公园长椅总藏着温柔的时光。阳光穿过疏朗的梧桐叶，在木质椅面上洒下细碎的金斑，"
# input_text = "守护文化遗产是人类对自身文明根脉的敬畏与担当。从故宫红墙下的斗拱飞檐到敦煌莫高窟的飞天壁画，"
# input_text = "算法公平性是人工智能技术健康发展的核心伦理基石，也是避免技术加剧社会偏见的关键防线。在人工智能"


inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids.to("cuda")
print(input_ids)
output = model.generate(
    input_ids=input_ids,
    do_sample=True,
    max_length=256,
    temperature=0.6,          # 保持一定随机性
    top_k=10,                 # 限定候选词为概率最高的 10
    top_p=0.9,                # nucleus sampling，累积概率阈值
    repetition_penalty=1.1,   # 惩罚重复，>1.0 就会抑制复读
    # no_repeat_ngram_size=5    # 避免出现相同的 3-gram 片段
)
print(output[0])
print(tokenizer.decode(output[0], skip_special_tokens=False))
