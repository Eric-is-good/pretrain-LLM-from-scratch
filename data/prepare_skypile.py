from data_process import DataProcess
import json
from transformers import LlamaTokenizer
import os
import numpy as np

# # 加载预训练模型的tokenizer
# tokenizer = LlamaTokenizer.from_pretrained("model/", use_fast=True)

class SkyDataProcess(DataProcess):
    def get_all_data_files(self, data_dir):
        # 获取所有json文件
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".jsonl")]
    
    def precess_one_file(self, data_path):
        # 每一行都是一个json对象，读取里面的text字段
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.readlines()
            sentences = [json.loads(line)["text"] for line in data]
            # 拼接句子
            output = self.concat_sentense_tokenize(sentences)
            length = len(output)
            # 转换为numpy数组
            output = self.convert_list_to_numpy(output)
            # 保存为npy文件
            npy_file_name = data_path.replace(".jsonl", "")
            np.save(npy_file_name, output)
            print(f"Save {npy_file_name} successfully. Total {length} sentences.")
            
if __name__ == "__main__":
    # 加载预训练模型的tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("model/", use_fast=True)
    # 创建数据处理对象
    data_process = SkyDataProcess(tokenizer)
    # 获取所有数据文件
    data_process.get_all_data_files("C:/Users/Eric/Downloads/1")
    # 处理所有数据文件
    data_process.process_all_files()
                