"""
    执行模型的训练与测试
        采用全部特征数据
"""
from models.Models import *

CLASS_1_FILE_PATH = './data/class_1.dat'
CLASS_2_FILE_PATH = './data/class_2.dat'
CLASS_3_FILE_PATH = './data/class_3.dat'
CLASS_4_FILE_PATH = './data/class_4.dat'


if __name__ == '__main__':
    print("class_1:")
    class_1(CLASS_1_FILE_PATH)
    print("\nclass_2:")
    class_2(CLASS_2_FILE_PATH)
    print("\nclass_3:")
    class_3(CLASS_3_FILE_PATH)
    print("\nclass_4:")
    class_4(CLASS_4_FILE_PATH)
    pass

