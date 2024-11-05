import sys
sys.path.append("/home/eric/llm/")

from model.configuration_holmesllm import HolmesLLMConfig
from transformers import LlamaForCausalLM
from model.modeling_holmesllm import HolmesLLMForCausalLM
import transformers
from transformers import LlamaTokenizer
import torch
from datasets import Dataset
import numpy as np
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments
from typing import Dict, Optional, Sequence, List
import deepspeed

# 将每个 .npy 文件加载为字典
def generate_data(file_paths):
    for path in file_paths:
        data = np.load(path, mmap_mode='r')
        for row in data:
            yield {"input_ids": row}

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  


@dataclass
class HolmesArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="save_model/")
    data_path: str = field(default="/root/autodl-tmp/",
                           metadata={"help": "Path to the training data."})
    data_cache_path: str = field(default="/root/autodl-tmp/cache",
                           metadata={"help": "Path to the training data cache."})
    output_dir: str = field(default="save_model/", metadata={"help": "Path to save the model."})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    group_by_modality_length: bool = field(default=False)
    

def train():
    global local_rank
    parser = transformers.HfArgumentParser(HolmesArguments)
    training_args = parser.parse_args_into_dataclasses()[0]
    local_rank = deepspeed.comm.get_local_rank()
    print(local_rank)

    model = HolmesLLMForCausalLM.from_pretrained(training_args.model_name_or_path,
                                                cache_dir=training_args.cache_dir)

    rank0_print(model)
    
    model.config.use_cache = False
    model.generate_labels = True
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 创建 Dataset 对象
    file_paths = [os.path.join(training_args.data_path, file) 
                  for file in os.listdir(training_args.data_path) if file.endswith(".npy")]
    dataset = Dataset.from_generator(lambda: generate_data(file_paths), cache_dir=training_args.data_cache_path)

    trainer = Trainer(
        model=model,  
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_state()
    
    safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()