from data_process import DataProcess
import json
from transformers import LlamaTokenizer
import os
import numpy as np
import mmap
from tqdm import tqdm
import time

# # 加载预训练模型的tokenizer
# tokenizer = LlamaTokenizer.from_pretrained("model/", use_fast=True)

class SkyDataProcess(DataProcess):
    def get_all_data_files(self, data_dir):
        # 获取所有json文件
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".jsonl")]
    
    @staticmethod
    def __amend_to_ndarray(sub_array: np.ndarray, array: np.ndarray | None = None) -> np.ndarray:
        if array is None:
            return sub_array.reshape(1, -1).astype(np.uint16)
        else:    
            return np.concatenate((array, sub_array.reshape(1, -1).astype(np.uint16)), axis=0)
    
    # @staticmethod    
    # def __count_lines(filename):
    #     """在不完全加载文件的情况下计算文件行数"""
    #     import time
    #     import mmap
    #     filename = "E:\\Projects\\HolmesLM\\dataset\\skypile\\2021-17_zh_middle_0002.jsonl"
    #     start_time = time.time()
    #     with open(filename, 'r') as f:
    #         buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    #         ret = buf.count(b'\n')
    #         end_time = time.time()
    #         print(f"Time taken to count lines in {filename}: {end_time - start_time} seconds")
    #         return ret
    
    def precess_one_file(self, data_path):
        # 每一行都是一个json对象，读取里面的text字段
        npy_file_name = data_path.replace(".jsonl", "")
        array = None
        with open(data_path, "r", encoding="utf-8") as f:
            current_tokens = []
            # for every sentence
            for line in tqdm(f,total=410000, desc=f"Processing {data_path}, 总数仅供参考"):
                sentence = json.loads(line)["text"]
                tokens = self.tokenize_sentense(sentence)
                # append to current_tokens
                if len(current_tokens) + len(tokens) > self.max_length:
                    # 如果超出，将当前行填充到 self.max_length 并存入结果列表
                    current_tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(current_tokens))
                    # 写入numpy数组
                    npy = self.convert_list_to_numpy(current_tokens)
                    array = self.__amend_to_ndarray(npy, array)
                    # 开始新行
                    current_tokens = tokens
                else:
                    # 否则将当前句子加入当前行
                    current_tokens += tokens
        # 最后一行填充到 self.max_length 并加入结果列表
        if current_tokens:
            current_tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(current_tokens))
            npy = self.convert_list_to_numpy(current_tokens)
            array = self.__amend_to_ndarray(npy, array)
        
        assert array is not None
        np.save(npy_file_name, array)
            
        print(f"Save {npy_file_name} successfully. Total {array.shape[0]} sentences.")
            
if __name__ == "__main__":
    dataset_dir = "E:\\Projects\\HolmesLM\\dataset\\skypile"
    # 加载预训练模型的tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("model/", use_fast=True)
    # 创建数据处理对象
    data_process = SkyDataProcess(tokenizer)
    # 获取所有数据文件
    data_process.get_all_data_files(dataset_dir)
    # 处理所有数据文件
    data_process.process_all_files()
                