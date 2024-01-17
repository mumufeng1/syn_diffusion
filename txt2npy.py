import numpy as np


def txt2npy(file_name):
    # ��.txt�ļ��ж�ȡÿ���ַ���
    with open(file_name+".txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ���������ַ���Ԫ�ص�NumPy����
    data = np.array([line.strip() for line in lines])

    # ��NumPy���鱣��Ϊ.npy�ļ�
    np.save(file_name + ".npy", data)


def npy2txt(file_name):
    # ��.npy�ļ��м�������
    data = np.load(file_name + ".npy", allow_pickle=True)

    # ��ÿ���ַ���Ԫ��д��.txt�ļ���ÿ��Ԫ��ռ��һ��
    with open(file_name+".txt", 'w', encoding='utf-8') as f:
        for string_element in data:
            f.write(str(string_element) + '\n')


