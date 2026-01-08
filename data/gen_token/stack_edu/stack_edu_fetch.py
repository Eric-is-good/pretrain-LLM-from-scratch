"""
Fatching files from stack edu
You should run stack_edu_sample first
"""
import os

import boto3
from tqdm import tqdm
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from datasets import load_from_disk

DATASET_DIR = '/home/users/nus/e1352689/scratch/stack_edu_sample'
N_THRED = 16


s3 = boto3.client('s3')
bucket_name = "softwareheritage"

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

def download_contents(blob_id, save_dir):
    key = f"content/{blob_id}"
    local_dir = os.path.join(save_dir, f"{blob_id}.gz")
    if os.path.exists(local_dir) or os.path.exists(local_dir+"failed"):
        return
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        with open(local_dir, 'wb') as f:
            f.write(obj['Body'].read())
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"File not found: {key}")
            with open(local_dir+"failed", 'w') as f:
                pass
        else:
            raise

for language in languages:
    print(f"Fetching {language}...")
    url_path = os.path.join(DATASET_DIR, language)
    download_path = os.path.join(DATASET_DIR, language+"_downloaded")
    if not os.path.exists(url_path):
        print(f"{url_path} not exist, skipping language {language}")
    os.makedirs(download_path, exist_ok=True)
    ds = load_from_disk(url_path)
    
    job = lambda x: download_contents(x['blob_id'], download_path)
    data_iter = iter(ds)
    with ThreadPoolExecutor(N_THRED) as executor:
        with tqdm(desc="Downloading...", total=len(ds)) as pbar:
            pending = set(executor.submit(job, next(data_iter)) for _ in range(N_THRED))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pbar.update(1)
                    future.result() # no return value, here will only through exception (if occured)
                    try:
                        pending.add(executor.submit(job, next(data_iter)))
                    except StopIteration:
                        continue
            