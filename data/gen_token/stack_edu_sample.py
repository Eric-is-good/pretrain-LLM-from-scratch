""" 
stack edu子集采样，获得约10b的子集
考虑到不是样本分数越高越好
(参考[except for Java, which performed best at threshold 2.](https://huggingface.co/datasets/HuggingFaceTB/stack-edu#dataset-curation))
这里使用随机采样
"""
import os

from datasets import load_dataset, Dataset
import numpy as np

RANDOM_SEED = 42
TARGET_SIZE_BILLION = 10 # (in billion)
BYTES_PER_TOKEN = 4.5 # (for holmes tokenizer, in avg)
SAVE_DIR = '/home/users/nus/e1352689/scratch/stack_edu_sample'
HF_CACHE_DIR = '/home/users/nus/e1352689/scratch/hf_cache' # make sure we don't use wrong space

language_subsets = {
    "Python": 21.8,
    "Cpp": 16.0,
    "Markdown": 14.0,
    "C": 11.1,
    "JavaScript": 11.1,
    "Java": 42.1,
    "SQL": 9.62,
    "PHP": 9.07,
    "CSharp": 8.87,
    "TypeScript": 3.03,
    "Shell": 3.13,
    "Swift": 1.83,
    "Go": 1.80,
    "Rust": 1.75,
    "Ruby": 1.61,
}

tot_token_billion = sum(v for v in language_subsets.values())

for language, size_billion in language_subsets.items():
    save_path = os.path.join(SAVE_DIR, language)
    if os.path.exists(save_path):
        print(f"{save_path} not empty, skipping language {language}")
        continue
    
    # 1. Acquire dataset
    print(f"Loading subset {language}")
    subset = load_dataset(
        'HuggingFaceTB/stack-edu',
        name = language,
        split='train',
        cache_dir=HF_CACHE_DIR
    )
    assert isinstance(subset, Dataset)
    
    size = len(subset)
    
    subset_ratio = size_billion / tot_token_billion
    token_goal_billion = subset_ratio * TARGET_SIZE_BILLION
    print(f"Sampling, target size {token_goal_billion}B")
    
    # rough range
    # take sampling ratio * 2 or full (if smaller)
    subset = subset.select(range(min(int(len(subset) * (TARGET_SIZE_BILLION/tot_token_billion) * 2), len(subset))))
    
    length = np.array(subset['length_bytes'])
    accum_length = np.cumsum(length)
    
    cutoff_idx = np.searchsorted(accum_length, token_goal_billion * BYTES_PER_TOKEN * 1e9, side='right') # 只大不小
    
    subset = subset.select(range(cutoff_idx))
    print(f"Sampled {cutoff_idx} of {size} samples, total {accum_length[cutoff_idx]} bytes")
    
    subset = subset.select_columns('blob_id') # we only need blob_id for download
    
    subset.save_to_disk(os.path.join(SAVE_DIR, language))
