import numpy as np
import sklearn.utils as su
from sklearn.preprocessing import scale


def load_data(file_path):
    """
        加载数据, 并拆分为训练集和测试集
    """
    data = []
    # 加载数据
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data.append(line.strip().split(','))
    data = np.array(data)
    # 拆分样本数据和标签
    x = data[1:, :-1].astype('f8')
    y = data[1:, -1].astype('f8')
    # 打乱数据集, 拆分测试集(20%)与训练集(80%)
    x, y = su.shuffle(x, y, random_state=7)
    # 标准化样本数据
    x_std = scale(x)
    train_size = int(len(x_std) * 0.75)
    # 拆分训练集/测试集
    train_x, test_x, train_y, test_y = x_std[:train_size], x_std[train_size:], y[:train_size], y[train_size:]
    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    pass
