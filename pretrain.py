import os
import math
import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import json
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    TrainerCallback,
    set_seed,
)
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.append("/home/zhupeiyin/code/pretrain-LLM-from-scratch/model/")

from configuration_holmes import HolmesConfig
from modeling_holmes import HolmesForCausalLM, HolmesMoE 

# ===== 路径配置 =====
PROCESSED_DATA_PATH = "/local_home/zhupeiyin/data/mini/processed_data"
TOKENIZER_PATH = "/home/zhupeiyin/code/pretrain-LLM-from-scratch/tokenizer"
CONFIG_PATH = "model/config.json"  # 你上一步生成的小配置

# ===== 训练超参（示例，可按需改）=====
BLOCK_SIZE = 4096
BATCH_SIZE_PER_DEVICE = 2          # 如果显存够，可调大
GRAD_ACCUM = 1                     # 全局有效 batch = BATCH_SIZE_PER_DEVICE * n_gpus * GRAD_ACCUM，TODO 由于有死锁bug，暂时只能 1
LEARNING_RATE = 5e-4               # 预训练常用 1e-4 ~ 3e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.1
NUM_TRAIN_EPOCHS = 1               # 或改为 max_steps（例如 50k/100k）
SAVE_STEPS = 1000
LOGGING_STEPS = 10
EVAL_STEPS = 0                     # 纯预训练可不评估，或做少量 held-out
MIN_LR_RATIO = 0.02                # 5e-4 -> 1e-5
SEED = 42

# ===== 自定义 DataCollator：只根据 input_ids 生成 labels =====
@dataclass
class SimpleCausalCollator:
    tokenizer: PreTrainedTokenizerBase
    block_size: int
    # 注意：我们这里假设已经是定长块，不做 padding，也不做 mask
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # features: [{'input_ids': [..BLOCK_SIZE..]}, ...]
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch = torch.stack(input_ids, dim=0)  # (B, L)
        # causal LM 通常 labels = input_ids（模型内部会 shift）
        return {"input_ids": batch, "labels": batch.clone()}


class MoELoadBalancingLogger(TrainerCallback):
    """
    一个自定义回调，用于记录 MoE 层的专家负载均衡指标。
    只在指定的 rank 上执行记录，以避免多卡写入冲突。
    """
    def __init__(self, log_file="moe_load_stats.jsonl", log_every_n_steps=10, log_from_rank=0):
        """
        增加 log_from_rank 参数，默认为 0 (主进程)。
        """
        self.log_file = log_file
        self.log_every_n_steps = log_every_n_steps
        self.log_from_rank = log_from_rank
        self._has_cleared_file = False # 用于确保文件只被清空一次

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        在每个训练步骤结束后被调用。
        """
        if args.process_index != self.log_from_rank:
            return

        # 确保日志文件只在第一次写入时被清空一次
        if not self._has_cleared_file:
            with open(self.log_file, "w") as f:
                pass # 创建一个空文件或清空已有文件
            self._has_cleared_file = True

        # 检查是否达到了记录的频率
        if state.global_step > 0 and state.global_step % self.log_every_n_steps == 0:
            if model is None:
                return

            step_stats = {"step": state.global_step}
            moe_layer_idx = 0
            for i, layer in enumerate(model.model.layers):
                if isinstance(layer.mlp, HolmesMoE):
                    if hasattr(layer.mlp.gate, 'last_batch_load'):
                        load_list = layer.mlp.gate.last_batch_load
                        step_stats[f"moe_layer_{i}"] = load_list
                        moe_layer_idx += 1
            
            if moe_layer_idx > 0:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(step_stats) + "\n")


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        重写学习率调度器，以支持带有 warmup 和最小学习率的余弦退火。
        """
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            
            num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
            
            # 定义学习率衰减的 lambda 函数
            def lr_lambda(current_step):
                # 1. Warmup 阶段：学习率从 0 线性增加到 1 (倍率)
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                
                # 2. Cosine Decay 阶段：学习率从 1 余弦衰减到 MIN_LR_RATIO
                # 首先，计算衰减阶段的进度，从 0 到 1
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                
                # 确保 progress 不会超出 [0, 1] 范围
                progress = min(max(0.0, progress), 1.0)

                # 计算余弦曲线上的值，从 1 变化到 0
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                
                # 将余弦值映射到 [MIN_LR_RATIO, 1.0] 的区间
                decayed_lr_scale = MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine_decay
                
                return decayed_lr_scale

            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            
        return self.lr_scheduler




def main():
    set_seed(SEED)

    # 1) 加载数据（只含 input_ids）
    ds = load_from_disk(PROCESSED_DATA_PATH)
    # 你可以把一小部分切出来做 eval，例如：
    # ds = ds.train_test_split(test_size=0.001, seed=SEED)
    # train_ds, eval_ds = ds["train"], ds["test"]
    train_ds, eval_ds = ds, None

    # 2) 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) 加载配置与模型
    config = HolmesConfig.from_pretrained(CONFIG_PATH)
    
    config._attn_implementation = "flash_attention_2"
    config.use_cache = False 
    
    # 建议 bfloat16（A100/H100/4090 支持），否则 fp16
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = HolmesForCausalLM(config).to(dtype=torch_dtype)

    # 性能/稳定性增强
    torch.backends.cuda.matmul.allow_tf32 = True
    model.gradient_checkpointing_enable()

    # 4) 构造 collator
    collator = SimpleCausalCollator(tokenizer=tokenizer, block_size=BLOCK_SIZE)

    # 5) 训练参数
    #   - 如果你要多卡，直接用 accelerate 或 torchrun 启动，Trainer 会自动处理 DDP
    args = TrainingArguments(
        deepspeed="ds_zero.json",
        output_dir="ds3_pretrain_ckpts",
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="no" if eval_ds is None else "steps",
        eval_steps=EVAL_STEPS if eval_ds is not None else None,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        torch_compile=False,  # PyTorch 2.0 编译，可能更快但不稳定可关
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to=["none"],             # 或 ["tensorboard"]
        dataloader_num_workers=min(8, os.cpu_count()//2 or 8),
        optim="adamw_torch_fused",      # PyTorch fused AdamW（CUDA≥11.7），不支持则自动回退
        max_grad_norm=1.0,
        save_total_limit=2
    )

    moe_logger = MoELoadBalancingLogger(
        log_file="ds3_pretrain_ckpts/moe_load_stats.jsonl", 
        log_every_n_steps=10,
        log_from_rank=3  # <--- 指定只有 rank 3 (通常是 GPU 3) 进行记录
    )
    
    # 6) Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=train_ds,
    #     eval_dataset=eval_ds,
    #     data_collator=collator,
    #     tokenizer=tokenizer,  # 主要用于保存
    #     callbacks=[moe_logger],
    # )
    trainer = CustomTrainer(        
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[moe_logger],
    )

    # 7) 开始训练
    trainer.train()

    # 8) 保存最终权重与 tokenizer、config
    trainer.save_model("ds3_pretrain_ckpts/final")
    tokenizer.save_pretrained("ds3_pretrain_ckpts/final")
    print("✅ 预训练完成，已保存到 ds3_pretrain_ckpts/final")

if __name__ == "__main__":
    main()
