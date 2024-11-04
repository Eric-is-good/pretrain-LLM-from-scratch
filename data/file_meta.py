import os

import pandas as pd
import numpy as np
from fire import Fire

def main(data_dir: str = "E:\\Projects\\HolmesLM\\dataset\\skypile", num_per_folder: int = 50, folder_start: int = 2):
    assert os.path.isdir(data_dir), "data_dir must be a directory"
    print("Start grouping files")
    file_list = [file for file in os.listdir(data_dir) if file.endswith(".npy")]
    grouped_files = [file_list[i:i + num_per_folder] for i in range(0, len(file_list), num_per_folder)]
    
    for group_index, group in enumerate(grouped_files):
        group_folder = os.path.join(data_dir, f"{folder_start + group_index}")
        print(f"Group {folder_start + group_index} with {len(group)} files")
        os.mkdir(group_folder)
        metadata = pd.DataFrame(columns=["file", "shape", "original_name"])
        for i, file in enumerate(group):
            new_name = f"{i}.npy"
            shape = np.load(os.path.join(data_dir, file)).shape
            metadata = pd.concat([metadata, pd.DataFrame([[new_name, shape, file.strip(".npy")]], columns=["file", "shape", "original_name"])])
            os.rename(os.path.join(data_dir, file), os.path.join(group_folder, new_name))
        metadata.to_csv(os.path.join(group_folder, "metadata.csv"), index=False)
        print(f"Group {folder_start + group_index} done")
        
if __name__ == "__main__":
    Fire(main)