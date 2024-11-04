import os
from concurrent.futures import ProcessPoolExecutor

import wget

FILE_DIR = "E:\\Projects\\HolmesLM\\dataset\\skypile"

def get_file(url):  
    file_name = url.strip().split("/")[-1]
    file_name = os.path.join(FILE_DIR, file_name)
    wget.download(url, file_name)

if __name__ == "__main__":
    
    with open("url2.txt", "r", encoding="utf-8") as f:
        urls = f.readlines()
        
    existed_files = [file for file in os.listdir(FILE_DIR) if file.endswith(".jsonl")]

    urls = [url.strip() for url in urls if url.strip().split("/")[-1] not in existed_files]
    total = len(urls)
    with ProcessPoolExecutor(max_workers=10) as executor:
        for i, _ in enumerate(executor.map(get_file, urls)):
            print(f"\n{i}/{total}")