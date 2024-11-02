from model.configuration_holmesllm import HolmesLLMConfig
from model.modeling_holmesllm import HolmesLLMForCausalLM
from transformers import LlamaTokenizer

config_path = "model/"

config = HolmesLLMConfig.from_pretrained(config_path, trust_remote_code=True)
model = HolmesLLMForCausalLM(config).to(dtype=config.torch_dtype)
tokenizer = LlamaTokenizer.from_pretrained(config_path, use_fast=True)

# 计算模型的参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# 测试模型架构，只需要 forward 成功即可
input_text = "who live under the sea?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids
print(input_ids)
output = model.generate(input_ids=input_ids, max_length=10)

print(output)

# from transformers import LlamaTokenizer

# # 加载预训练模型的tokenizer
# tokenizer = LlamaTokenizer.from_pretrained("model/")

# # 把pad_token设置为<pad>
# tokenizer.pad_token = "<pad>"

# print(tokenizer.pad_token)
# print(tokenizer.pad_token_id)

# # 保存新的tokenizer
# tokenizer.save_pretrained("t/")