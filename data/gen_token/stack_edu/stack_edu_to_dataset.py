import os
import gzip
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

LOAD_DIR = '/home/users/nus/e1352689/scratch/stack_edu_sample'
SAVE_DIR = '/home/users/nus/e1352689/scratch/stack_edu_subset'
HF_CACHE_DIR = '/home/users/nus/e1352689/scratch/hf_cache' # make sure we don't use wrong space
N_THRED = 32

os.environ['HF_HOME'] = HF_CACHE_DIR
from datasets import Dataset

languages = [
    "Python",
    "Cpp",
    "Markdown",
    "C",
    "JavaScript",
    "Java",
    "SQL",
    "PHP",
    "CSharp",
    "TypeScript",
    "Shell",
    "Swift",
    "Go",
    "Rust",
    "Ruby"
]

def open_file(file_path):
    if file_path.endswith('failed'):
        blob_id = os.path.split(file_path)[-1].rstrip('.gzfailed')
        return {'blob_id': blob_id, 'content': None}
    elif file_path.endswith('.gz'):
        blob_id = os.path.split(file_path)[-1].rstrip('.gz')
        with gzip.GzipFile(file_path) as fin:
            content = fin.read().decode("utf-8", errors="ignore")
            return {"blob_id": blob_id, "content": content}
    else:
        raise ValueError(f"Invalid file path {file_path}")

def get_iterator(path: str):
    def iterator():
        with ProcessPoolExecutor(max_workers=N_THRED) as executor:
            path_iter = iter(entry.path for entry in os.scandir(path) if entry.is_file() and (entry.path.endswith(".gz") or entry.path.endswith(".gzfailed")))
            pending = set(executor.submit(open_file, next(path_iter)) for _ in range(N_THRED))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    try:
                        pending.add(executor.submit(open_file, next(path_iter)))
                    except StopIteration:
                        continue
                
    return iterator

for language in languages:
    print(f"Converting {language}...")
    load_path = os.path.join(LOAD_DIR, language+"_downloaded")
    save_path = os.path.join(SAVE_DIR, language)
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping..")
        continue
    dataset = Dataset.from_generator(get_iterator(load_path), cache_dir=HF_CACHE_DIR)
    dataset.save_to_disk(save_path)
    
    