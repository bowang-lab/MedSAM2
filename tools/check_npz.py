import numpy as np
import sys
import os


def print_npz_keys(file_path):
    """
    加载一个NPZ文件并打印出其中包含的所有键。

    参数:
    file_path (str): NPZ文件的路径。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到 -> {file_path}")
        return

    # 检查文件扩展名是否为 .npz
    if not file_path.endswith('.npz'):
        print(f"警告: 文件 '{os.path.basename(file_path)}' 可能不是一个NPZ文件。")

    try:
        # 使用 np.load 加载文件
        with np.load(file_path) as data:
            # .files 属性会返回一个包含所有键的列表
            keys = data.files

            print(f"文件 '{os.path.basename(file_path)}' 中包含的键 (Keys):")

            if not keys:
                print("  -> 此文件为空，不包含任何数组。")
            else:
                # 逐行打印每个键
                for key in keys:
                    print(f"  - {key}")

    except Exception as e:
        print(f"错误: 加载或读取文件时发生错误 -> {file_path}")
        print(f"  详细信息: {e}")


if __name__ == '__main__':
    npz_file_path = sys.argv[1]
    print_npz_keys(npz_file_path)
