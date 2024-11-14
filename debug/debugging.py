import numpy as np
from transformers import LlamaTokenizer

if __name__ == "__main__":
    data = np.load("2020-40_zh_head_0000.npy")
    start_step = 890
    start = 890 * 64
    end = start + 64 * 10
    problemetic_data = data[start:end]
    tokenizer = LlamaTokenizer.from_pretrained("../model/", use_fast=True)
    lines = [tokenizer.decode(l) for l in problemetic_data]
    with open("problematic.txt", "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            f.write(f"======line: {start+i}======\n")
            f.write(line + "\n")