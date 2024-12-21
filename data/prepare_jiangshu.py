import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import LlamaTokenizer
from datasets import Dataset, concatenate_datasets

from data_process import DataProcess


class JiangshuDataProcess(DataProcess):
    @staticmethod
    def get_processed_files(data_dir: str) -> list:
        processed_files = set(file for file in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, file)))
        return list(processed_files)
    
    def get_all_data_files(self, data_dir):
        # 获取所有json文件
        processed_files = self.get_processed_files(data_dir)
        all_files = os.listdir(data_dir)
        self.data_files = [os.path.join(data_dir, file) for file in all_files if file.endswith(".jsonl") and file.strip(".jsonl") not in processed_files]
        print(f"{len(self.data_files)} files to process. Total {len(self.data_files) + len(processed_files)} files.")
    
    
    def process_one_file(self, data_path, context=200):
        def process_json_to_messages(data):
            messages = []

            # Extract history
            for history_item in data.get('history', []):
                messages.append({"role": "user", "content": str(history_item[0])})
                messages.append({"role": "assistant", "content": str(history_item[1])})

            # Extract input and output
            input_content = data.get('instruction', '') + data.get('input', '')
            output_content = data.get('output', '')

            if input_content:
                messages.append({"role": "user", "content": str(input_content)})
            if output_content:
                messages.append({"role": "assistant", "content": str(output_content)})

            return messages
        
        try:

            # 每一行都是一个json对象，读取里面的text字段
            array = []
            print(data_path)
            with open(data_path, "r", encoding="utf-8") as f:
                # for every sentence
                for line in tqdm(f):
                    # sentence = json.loads(line)["text"]
                    json_sentence = json.loads(line)
                    sentence = process_json_to_messages(json_sentence) 
                    # print(sentence)               
                    
                    tokens = self.tokenize_conversation(sentence)

                    if len(tokens) > self.max_length:
                        npy = self.convert_list_to_numpy(tokens[:self.max_length])
                        with open("long.txt", "a", encoding="utf-8") as long_f:
                            long_f.write(str(line))
                    else:
                        tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
                        npy = self.convert_list_to_numpy(tokens)
                    
                    array.append({"input_ids": npy})

                        
            # 取data_path除开文件后缀的部分作为保存路径
            path = data_path.split(".")[0]
            print("saving to", path)
            dataset = Dataset.from_list(array)
            dataset.save_to_disk(path)
        except Exception as e:
            print(e)

if __name__ == "__main__":    
    dataset_dir = "/home/eric1234567812345678/code/pretrain-LLM-from-scratch/"
    folder = "jiangshu"
    # 加载预训练模型的tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("model/", use_fast=True)
    # 创建数据处理对象
    data_process = JiangshuDataProcess(tokenizer, max_length=4096)
    # 获取所有数据文件
    data_process.get_all_data_files(os.path.join(dataset_dir, folder))
    # 处理所有数据文件
    data_process.process_all_files(4)
    # conbine datasets
    dataset_dirs = [os.path.join(dataset_dir, folder, dataset) for dataset in  data_process.get_processed_files(os.path.join(dataset_dir, folder))]
    concatenate_datasets([Dataset.load_from_disk(dataset) for dataset in dataset_dirs]).save_to_disk(os.path.join(dataset_dir, f"{folder}_dataset"))
           
