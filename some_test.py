#################################### test model #####################################
# model.modeling_holmesllm.py 中 687 行写死了 eager 用于推理, 需要 flash_attn 则注释掉
from model.modeling_holmesllm import HolmesLLMForCausalLM
from transformers import LlamaTokenizer

model = HolmesLLMForCausalLM.from_pretrained("model/").to("cuda")
tokenizer = LlamaTokenizer.from_pretrained("model/")
model.eval()

input_text = "<|startoftext|>菠萝和凤梨的区别是"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids.to("cuda")
print(input_ids)
output = model.generate(input_ids=input_ids, max_length=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1)
print(output[0])
print(tokenizer.decode(output[0], skip_special_tokens=False))
