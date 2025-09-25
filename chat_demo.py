from transformers import AutoTokenizer

import sys
sys.path.append("/home/zhupeiyin/code/pretrain-LLM-from-scratch/model/")  # 这个需要更改
from modeling_holmes import HolmesForCausalLM

model = HolmesForCausalLM.from_pretrained("/home/zhupeiyin/code/pretrain-LLM-from-scratch/final/").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("/home/zhupeiyin/code/pretrain-LLM-from-scratch/final/")
model.eval()

input_text = "近年来，人工智能的发展出现了"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids.to("cuda")
print(input_ids)
output = model.generate(
    input_ids=input_ids,
    do_sample=True,
    max_length=256,
    temperature=1.0,          # 保持一定随机性
    top_k=25,                 # 限定候选词为概率最高的 50
    top_p=0.9,                # nucleus sampling，累积概率阈值
    repetition_penalty=1.2,   # 惩罚重复，>1.0 就会抑制复读
    # no_repeat_ngram_size=5    # 避免出现相同的 n-gram 片段
)
print(output[0])
print(tokenizer.decode(output[0], skip_special_tokens=False))
