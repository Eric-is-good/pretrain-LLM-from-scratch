import os
from concurrent.futures import ThreadPoolExecutor

import wget

FILE_DIR = "E:\\Projects\\HolmesLM\\dataset\\skypile"

def get_file(url):  
    file_name = url.strip().split("/")[-1]
    file_name = os.path.join(FILE_DIR, file_name)
    wget.download(url, file_name)

if __name__ == "__main__":
    
    with open("data\\skypile\\url2.txt", "r", encoding="utf-8") as f:
        urls = f.readlines()
        
    existed_files = [file for file in os.listdir(FILE_DIR) if file.endswith(".jsonl")]

    urls = [url.strip() for url in urls if url.strip().split("/")[-1] not in existed_files]
    total = len(urls)
    with ThreadPoolExecutor() as executor:
        for i, _ in enumerate(executor.map(get_file, urls)):
            print(f"\n{i+1}/{total}")
    
    # with open("data\\skypile\\url2.txt", "a", encoding="utf-8") as f:
    #     for i in range(7):
    #         f.write(f"https://huggingface.co/datasets/Skywork/SkyPile-150B/resolve/main/data/2020-50_zh_head_00{i:02d}.jsonl\n")