from models.Preprocessing import DataPreprocessing


if __name__ == '__main__':
    # 根据不同并发症拆分训练样本, 并保存至本地文件
    file_path = "./data/data_1234.csv"
    dpf = DataPreprocessing(file_path)
    dpf.run()


