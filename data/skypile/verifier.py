import numpy as np

def verify(file_1, file_2):
    array1 = np.load(file_1)
    array2 = np.load(file_2)
    assert array1.shape == array2.shape
    assert np.all(array1 == array2)
    print("Array in files are the same.")
    
if __name__ == "__main__":
    file1 = "E:\\Projects\\HolmesLM\\dataset\\skypile\\2021-17_zh_middle_0002.npy"
    file2 = "E:\\Projects\\HolmesLM\\dataset\\skypile\\2021-17_zh_middle_0002.npy"
    verify(file1, file2)