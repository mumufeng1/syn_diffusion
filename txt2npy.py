import numpy as np


def txt2npy(file_name):
    # 从.txt文件中读取每行字符串
    with open(file_name+".txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 创建包含字符串元素的NumPy数组
    data = np.array([line.strip() for line in lines])

    # 将NumPy数组保存为.npy文件
    np.save(file_name + ".npy", data)


def npy2txt(file_name):
    # 从.npy文件中加载数据
    data = np.load(file_name + ".npy", allow_pickle=True)

    # 将每个字符串元素写入.txt文件，每个元素占据一行
    with open(file_name+".txt", 'w', encoding='utf-8') as f:
        for string_element in data:
            f.write(str(string_element) + '\n')


