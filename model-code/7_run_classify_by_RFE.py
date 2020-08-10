"""
    执行模型的训练与测试
        采用特征消除(RFE)之后的特征数据
"""
from models.Models import *

CLASS_1_FILE_PATH = './data/RFE_class_1.dat'
CLASS_2_FILE_PATH = './data/RFE_class_2.dat'
CLASS_3_FILE_PATH = './data/RFE_class_3.dat'
CLASS_4_FILE_PATH = './data/RFE_class_4.dat'


if __name__ == '__main__':
    """
        注释部分为最优模型
    """
    print("class_1:")
    class_1(CLASS_1_FILE_PATH, data_type='RFE')  # svm_rbf_classifier
    print("\nclass_2:")
    class_2(CLASS_2_FILE_PATH, data_type='RFE')  # svm_linear_classifier
    print("\nclass_3:")
    class_3(CLASS_3_FILE_PATH, data_type='RFE')  # lr_classifier
    print("\nclass_4:")
    class_4(CLASS_4_FILE_PATH, data_type='RFE')  # svm_poly_classifier
    pass



