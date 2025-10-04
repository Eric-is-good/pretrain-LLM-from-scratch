import os
from datasets import load_from_disk
from itertools import chain
import logging

# --- 1. 配置参数 ---

# 输入：你已经处理好的、需要被重新分块的数据集路径
# 这个路径应该是你之前脚本中 `PROCESSED_DATA_PATH` 的值
INPUT_DATA_PATH = "/home/users/nus/e1352533/mydata/trans_p/processed"

# 输出：重新分块后，新数据集的保存路径
# 建议起一个能体现新块大小的名字
OUTPUT_DATA_PATH = "/home/users/nus/e1352533/mydata/trans_p/processed_2"

# 新的目标块大小
NEW_BLOCK_SIZE = 3072

# 使用的进程数，可以根据你的机器配置调整
NUM_PROC = os.cpu_count() // 2 or 8

# 设置日志级别，方便观察 datasets 库的内部信息
# logging.basicConfig(level=logging.INFO)


def reblock_dataset():
    """
    加载一个已处理的数据集，并将其重新分块为指定的新长度。
    """
    print(f"=== 步骤 1/4: 从磁盘加载已处理的数据集 ===")
    print(f"  -> 输入路径: {INPUT_DATA_PATH}")

    if not os.path.exists(INPUT_DATA_PATH):
        print(f"错误：在路径 '{INPUT_DATA_PATH}' 下找不到数据集。请检查路径是否正确。")
        return

    # 加载数据集
    # 这个数据集应该只包含 'input_ids' 列
    original_ds = load_from_disk(INPUT_DATA_PATH)

    print(f"加载成功！")
    print(f"  -> 原始样本数（旧 block 数）: {len(original_ds)}")
    print(f"  -> 数据集特征: {original_ds.features}")

    if "input_ids" not in original_ds.column_names:
        print("错误：数据集中未找到 'input_ids' 列。请确保加载了正确的数据。")
        return

    print(f"\n=== 步骤 2/4: 定义重新分块函数 (目标长度 {NEW_BLOCK_SIZE}) ===")

    # 这个函数的核心逻辑与你原始脚本中的 `group_texts` 类似
    def regroup_texts(examples):
        # `examples["input_ids"]` 是一个列表的列表，例如 [[id1, id2, ...], [id10, id11, ...]]
        # 使用 chain.from_iterable 将其“压平”成一个单一的长列表
        concatenated_ids = list(chain.from_iterable(examples["input_ids"]))
        total_length = len(concatenated_ids)

        # 如果连接后的总长度还不足一个新的块，则这个批次不产生任何输出
        if total_length < NEW_BLOCK_SIZE:
            return {"input_ids": []}

        # 我们只保留可以被 NEW_BLOCK_SIZE 整除的部分，丢弃末尾不足一个块的“零头”
        # 注意：这会导致在每个批次(batch)的边界处丢失少量数据。
        # 增大 batch_size 可以减少这种损失发生的频率。
        total_length = (total_length // NEW_BLOCK_SIZE) * NEW_BLOCK_SIZE

        # 将长列表切分成多个固定长度为 NEW_BLOCK_SIZE 的块
        result = [
            concatenated_ids[i : i + NEW_BLOCK_SIZE]
            for i in range(0, total_length, NEW_BLOCK_SIZE)
        ]
        return {"input_ids": result}

    print(f"  -> 函数定义完成。")

    print(f"\n=== 步骤 3/4: 应用 .map() 执行重新分块 ===")

    # 使用 .map() 来应用分块逻辑
    # batched=True 表示 regroup_texts 函数一次会收到一批数据
    # batch_size 控制每个批次包含多少个原始样本（块）。
    # 较大的 batch_size 会增加内存消耗，但可以减少因批次边界而导致的数据丢失。
    reblocked_ds = original_ds.map(
        regroup_texts,
        batched=True,
        batch_size=2000,  # 可根据内存大小调整，例如 1000, 2000
        num_proc=NUM_PROC,
        remove_columns=original_ds.column_names, # 移除旧列，只保留函数返回的新列
        desc=f"Grouping into blocks of {NEW_BLOCK_SIZE}",
    )

    final_count = len(reblocked_ds)
    if final_count == 0:
        print("\n警告：重新分块后没有生成任何样本。可能原因：")
        print(f"1. 原始数据总量太少，不足以构成一个 {NEW_BLOCK_SIZE} 的块。")
        print("2. `batch_size` 设置过小，导致每个批次拼接后都因长度不足而被丢弃。")
    else:
        print(f"\n重新分块完成！")
        print(f"  -> 最终样本数（新 block 数）: {final_count}")


    print(f"\n=== 步骤 4/4: 保存新的数据集到磁盘 ===")
    print(f"  -> 输出路径: {OUTPUT_DATA_PATH}")
    # os.makedirs(OUTPUT_DATA_PATH, exist_ok=True) # 确保目录存在
    reblocked_ds.save_to_disk(OUTPUT_DATA_PATH)
    print("\n✅ 所有操作完成，新的数据集已成功保存！")


if __name__ == "__main__":
    reblock_dataset()