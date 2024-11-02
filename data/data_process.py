from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

class DataProcess:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 4096
        self.data_files = []
    
    def get_all_data_files(self, data_dir):
        raise NotImplementedError
    
    def precess_one_file(self, data_path):
        raise NotImplementedError
    
    def process_all_files(self):
        # 使用多进程处理文件
        results = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            # 提交所有文件处理任务到进程池
            futures = {executor.submit(self.precess_one_file, file): file for file in self.data_files}
            
            # 收集处理结果
            for future in as_completed(futures):
                file = futures[future]
                try:
                    result = future.result()  # 获取处理结果
                    results.append((file, result))
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
        print("All files processed.")
                    
    
    def concat_sentense_tokenize(self, sentences):
        start_token = self.tokenizer.bos_token
        end_token = self.tokenizer.eos_token
        # 存储拼接后的句子
        concatenated_sentences = []
        current_tokens = []

        # 循环遍历每个句子，并按长度限制拼接
        for sentence in tqdm(sentences):
            # 为句子添加特殊标记
            sentence_with_tokens = f"{start_token} {sentence} {end_token}"
            
            # 将句子转换为 token ids
            tokens = self.tokenizer.encode(sentence_with_tokens, add_special_tokens=False, 
                                    truncation=True, max_length=self.max_length)
            
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


