from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class DataProcess:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 4096
        self.data_files = []
    
    def get_all_data_files(self, data_dir):
        raise NotImplementedError
    
    def precess_one_file(self, data_path):
        raise NotImplementedError
    
    def process_all_files(self, max_workers=6):
        # self.precess_one_file(self.data_files[0])
        # 使用多进程处理文件
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有文件处理任务到进程池
            executor.map(self.precess_one_file, self.data_files)
            # 等待所有任务完成
        print("All files processed.")
                    
    
    def tokenize_sentense(self, sentence):
        start_token = self.tokenizer.bos_token
        end_token = self.tokenizer.eos_token

        sentence_with_tokens = f"{start_token} {sentence} {end_token}"
            
        # 将句子转换为 token ids
        tokens = self.tokenizer.encode(sentence_with_tokens, add_special_tokens=False, 
                                truncation=True, max_length=self.max_length)
            
        return tokens
    
    def concat_sentense_tokenize(self, sentences):
        # 存储拼接后的句子
        concatenated_sentences = []
        current_tokens = []

        # 循环遍历每个句子，并按长度限制拼接
        for sentence in tqdm(sentences):
            tokens = self.tokenize_sentense(sentence)
            
            # 判断加入当前句子是否超出 self.max_length
            if len(current_tokens) + len(tokens) > self.max_length:
                # 如果超出，将当前行填充到 self.max_length 并存入结果列表
                current_tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(current_tokens))
                concatenated_sentences.append(current_tokens)
                # 开始新行
                current_tokens = tokens
            else:
                # 否则将当前句子加入当前行
                current_tokens += tokens

        # 最后一行填充到 self.max_length 并加入结果列表
        if current_tokens:
            current_tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(current_tokens))
            concatenated_sentences.append(current_tokens)
            
        return concatenated_sentences


    def convert_list_to_numpy(self, input_list, dtype=np.uint16):
        # 所有数字均在 0-65535 之间
        return np.array(input_list, dtype=dtype)


