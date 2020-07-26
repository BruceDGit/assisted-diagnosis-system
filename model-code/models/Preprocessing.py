import pandas as pd
import sklearn.preprocessing as sp


class DataPreprocessing:
    """
    将数据按照标签类别单独拆分出来，并保存成4份数据文件到data目录下
    """
    def __init__(self, file_path="./data/data_1234.csv"):
        self.RAW_FILE_PATH = file_path  # 原始文件路径
        self.CLASS_1_FILE_PATH = './data/class_1.dat'
        self.CLASS_2_FILE_PATH = './data/class_2.dat'
        self.CLASS_3_FILE_PATH = './data/class_3.dat'
        self.CLASS_4_FILE_PATH = './data/class_4.dat'
        self.raw_data = None
        self.data_sets = None
        self.__get_data_sets()

    def __get_data_sets(self):
        """
            处理样本数据集(只包含样本的指标数据)
        """
        self.raw_data = pd.read_csv(self.RAW_FILE_PATH)
        self.raw_data = self.raw_data.drop_duplicates()
        temp_data = self.raw_data.iloc[:, :-1]
        self.data_sets = temp_data.drop_duplicates()

    def __generate_train_sets(self, tag_value):
        """
            1.构造训练样本集, 重新打标签
                并发症发病的样本, 标签标记为1
                并发症未发病的样本, 标签标记为0
            2.一次操作只针对一种并发症, 并保存为一份训练样本
        """
        train_sets = []
        # 对病例样本逐个验证是否有相应并发症
        for data in self.data_sets.iterrows():
            line = ''
            for row in self.raw_data.iterrows():
                # 如果指标值完全一致, 则认定为同一个病例的数据
                if data[1].equals(row[1][:-1]) and row[1][-1] == tag_value:
                    temp_data = data[1].astype(dtype='str')
                    temp_list = temp_data.tolist()
                    temp_list.append('1')
                    line = ','.join(temp_list)
                    break
            if line:
                train_sets.append(line)
            else:
                temp_data = data[1].astype(dtype='str')
                temp_list = temp_data.tolist()
                temp_list.append('0')
                line = ','.join(temp_list)
                train_sets.append(line)
        return train_sets

    def __save_file(self, train_sets, class_file_path):
        """
            保存训练样本到文件
        """
        with open(class_file_path, 'w', encoding='utf-8') as f:
            f.write(','.join(list(self.raw_data.columns)))
            for line in train_sets:
                f.write('\n')
                f.write(line)

    def run(self):
        """
            主方法
        """
        class_1_train_sets = self.__generate_train_sets(1)
        self.__save_file(class_1_train_sets, self.CLASS_1_FILE_PATH)
        class_2_train_sets = self.__generate_train_sets(2)
        self.__save_file(class_2_train_sets, self.CLASS_2_FILE_PATH)
        class_3_train_sets = self.__generate_train_sets(3)
        self.__save_file(class_3_train_sets, self.CLASS_3_FILE_PATH)
        class_4_train_sets = self.__generate_train_sets(4)
        self.__save_file(class_4_train_sets, self.CLASS_4_FILE_PATH)


if __name__ == '__main__':
    # 根据不同并发症拆分训练样本, 并保存至本地文件
    file_path = "../data/data_1234.csv"
    dpf = DataPreprocessing(file_path)
    dpf.run()
