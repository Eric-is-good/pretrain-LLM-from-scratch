import torch
import torch.nn as nn
import json
import math
import warnings
from typing import List, Optional, Tuple, Union
from model.configuration_holmes import HolmesConfig
from model.modeling_holmes import HolmesForCausalLM, HolmesMLP, HolmesMoE



def create_small_config_file():
    """创建一个小型的模型配置文件 config_small.json"""
    config_small_dict = {
      "architectures": ["HolmesForCausalLM"],
      "model_type": "holmes",
      "initializer_range": 0.05,
      "hidden_size": 1536,
      "intermediate_size": 4096,
      "num_hidden_layers": 12,
      "num_attention_heads": 12,
      "num_key_value_heads": 12,
      "v_head_dim": 128, # hidden_size / num_attention_heads
      "qk_nope_head_dim": 128,
      "qk_rope_head_dim": 16,
      "q_lora_rank": 384,
      "kv_lora_rank": 192,
      "moe_layer_freq": 1,
      "first_k_dense_replace": 2,
      "n_routed_experts": 48,
      "n_shared_experts": 1,
      "num_experts_per_tok": 3,
      "moe_intermediate_size": 320,
      "n_group": 1,
      "topk_group": 1,
      "vocab_size": 73448,
      "max_position_embeddings": 4096,
      "hidden_act": "silu",
      "rms_norm_eps": 1e-06,
      "rope_theta": 10000.0,
      "attention_bias": False,
      "attention_dropout": 0.0,
      "topk_method": "noaux_tc",
      "scoring_func": "sigmoid",
      "norm_topk_prob": True,
      "routed_scaling_factor": 2.5,
      "use_cache": True,
      "bos_token_id": 1,
      "eos_token_id": [2, 73440],
      "torch_dtype": "bfloat16",
      "tie_word_embeddings": True
    }
    with open("config_small.json", "w") as f:
        json.dump(config_small_dict, f, indent=2)
    print("已生成小型配置文件: config_small.json")


def calculate_params(model, config):
    """计算模型的总参数和激活参数"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 由于 tie_word_embeddings=True, lm_head 和 embed_tokens 共享权重, 需要从总数中减去一份 lm_head 的参数
    # Hugging Face 的 PreTrainedModel 会自动处理权重绑定，这里的计算方式是为了验证
    if config.tie_word_embeddings:
        total_params -= sum(p.numel() for p in model.lm_head.parameters())
    
    # 初始化非 MoE 层的参数计数器
    non_moe_params = 0
    # MoE 层中所有路由专家的总参数
    total_routed_expert_params = 0
    # MoE 层中共享专家的总参数
    total_shared_expert_params = 0
    
    # 遍历所有层来区分参数类型
    for layer in model.model.layers:
        # 累加注意力层和归一化层的参数
        non_moe_params += sum(p.numel() for p in layer.self_attn.parameters())
        non_moe_params += sum(p.numel() for p in layer.input_layernorm.parameters())
        non_moe_params += sum(p.numel() for p in layer.post_attention_layernorm.parameters())

        # 判断是 MoE 层还是密集 MLP 层
        if isinstance(layer.mlp, HolmesMoE):
            # 累加门控网络的参数
            non_moe_params += sum(p.numel() for p in layer.mlp.gate.parameters())
            # 累加共享专家的参数
            if layer.mlp.shared_experts is not None:
                total_shared_expert_params += sum(p.numel() for p in layer.mlp.shared_experts.parameters())
            # 累加所有路由专家的参数 (在 ep_size=1 时, 所有专家都非 None)
            for expert in layer.mlp.experts:
                if expert is not None:
                    total_routed_expert_params += sum(p.numel() for p in expert.parameters())

        elif isinstance(layer.mlp, HolmesMLP):
            # 这是常规的密集层，所有参数都算作非 MoE 参数
            non_moe_params += sum(p.numel() for p in layer.mlp.parameters())

    # 加上 embedding 和 final norm 层的参数
    non_moe_params += sum(p.numel() for p in model.model.embed_tokens.parameters())
    non_moe_params += sum(p.numel() for p in model.model.norm.parameters())
    
    # 计算单个路由专家的参数量
    params_per_routed_expert = 0
    for layer in model.model.layers:
        if isinstance(layer.mlp, HolmesMoE):
            for expert in layer.mlp.experts:
                if expert is not None:
                    params_per_routed_expert = sum(p.numel() for p in expert.parameters())
                    break
            break # 找到第一个 MoE 层后就停止

    # 计算激活的路由专家参数量
    num_moe_layers = sum(1 for layer in model.model.layers if isinstance(layer.mlp, HolmesMoE))
    activated_routed_expert_params = num_moe_layers * params_per_routed_expert * config.num_experts_per_tok
    
    # 激活参数 = 非MoE参数 + 共享专家参数 + 激活的路由专家参数
    activated_params = non_moe_params + total_shared_expert_params + activated_routed_expert_params
    
    return total_params, activated_params


def run_forward_and_backward_pass(model, config):
    """执行一次完整的前向传播、损失计算和反向传播测试"""
    print("\n--- 开始前向和反向传播测试 ---")
    
    # 设置模型为训练模式
    model.train().cuda()
    
    # 创建虚拟输入数据
    batch_size = 2
    seq_length = 64
    dummy_input_ids = torch.randint(
        0, config.vocab_size, 
        (batch_size, seq_length), 
        dtype=torch.long
    ).cuda()
    dummy_attention_mask = torch.ones(
        (batch_size, seq_length), 
        dtype=torch.long
    ).cuda()
    # 创建虚拟标签，这里为了方便直接使用 input_ids
    # 模型内部会自动移位(shift)来计算损失，所以可以直接使用
    dummy_labels = dummy_input_ids.clone()
    
    print(f"输入尺寸 (input_ids): {dummy_input_ids.shape}")
    print(f"输入尺寸 (labels): {dummy_labels.shape}")
    
    try:
        # 清零之前的梯度
        model.zero_grad()
        
        # 执行前向传播
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            labels=dummy_labels
        )
        
        # 获取损失和 logits
        loss = outputs.loss
        logits = outputs.logits
        
        print(f"✅ 前向传播成功！")
        print(f"输出 Logits 尺寸: {logits.shape}")
        print(f"计算出的损失 (Loss): {loss.item()}")
        
        # 预期输出尺寸: (batch_size, seq_length, vocab_size)
        expected_shape = (batch_size, seq_length, config.vocab_size)
        assert logits.shape == expected_shape, f"输出尺寸错误！预期: {expected_shape}, 得到: {logits.shape}"
        print("✅ 输出尺寸正确！")

        # 执行反向传播
        loss.backward()
        print("✅ 反向传播成功！")

        # 检查一个参数是否有梯度，以验证反向传播过程
        # 例如，检查 lm_head 的权重梯度
        if model.lm_head.weight.grad is not None:
            print("✅ 梯度检查通过 (lm_head.weight.grad is not None)。")
        else:
            print("❌ 梯度检查失败 (lm_head.weight.grad is None)。")

    except Exception as e:
        print(f"❌ 前向或反向传播失败！")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. 创建小型配置文件
    create_small_config_file()
    
    # 2. 从配置文件初始化模型
    print("\n--- 初始化模型 ---")
    config = HolmesConfig.from_pretrained("config_small.json")
    # 为了梯度计算，将模型转换为 float32
    model = HolmesForCausalLM(config).to(dtype=torch.float32).cuda()
    print("模型初始化成功！")
    
    # 3. 计算并打印参数量
    print("\n--- 计算参数量 ---")
    total, activated = calculate_params(model, config)
    print(f"模型的总参数量: {total / 1e6:.2f} M")
    print(f"模型的激活参数量: {activated / 1e6:.2f} M")
    
    # 4. 执行前向和反向传播测试
    run_forward_and_backward_pass(model, config)