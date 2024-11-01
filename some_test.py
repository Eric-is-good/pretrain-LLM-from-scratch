from model.configuration_holmesllm import HolmesLLMConfig
from model.modeling_holmesllm import HolmesLLMForCausalLM
from transformers import AutoTokenizer

config_path = "model/"

config = HolmesLLMConfig.from_pretrained(config_path, trust_remote_code=True)
model = HolmesLLMForCausalLM(config).to(dtype=config.torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(config_path, use_fast=True)

# 计算模型的参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# 测试模型架构，只需要 forward 成功即可
input_text = "Hello, my dog is cute"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids
output = model.generate(input_ids=input_ids, max_length=10)

print(output)
