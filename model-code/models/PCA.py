import pandas as pd
from sklearn.decomposition import PCA
import sklearn.preprocessing as sp

inputfile_1 = "./data/class_1.dat"
inputfile_2 = "./data/class_2.dat"
inputfile_3 = "./data/class_3.dat"
inputfile_4 = "./data/class_4.dat"
outputfile_1 = "./data/new_class_1.dat"
outputfile_2 = "./data/new_class_2.dat"
outputfile_3 = "./data/new_class_3.dat"
outputfile_4 = "./data/new_class_4.dat"


def get_pca_tk(data_file):
    data = pd.read_csv(data_file)

    # 提取特征名
    features = data.columns
    features = list(features[:-1])

    # 对特征数据标准化处理
    std_samples = sp.scale(data.iloc[:, :-1])  # ndarray类型
    # ndarray 转 dataframe类型
    std_samples = pd.DataFrame(std_samples)

    # 主成分分析
    pca = PCA()
    pca.fit(std_samples)
    # 返回各个成分各自的方差百分比(贡献率)
    Contribution_rate = pca.explained_variance_ratio_
    # 得到贡献率字典
    tk_dict = { feature: round(item, 4) for item, feature in zip(Contribution_rate, features)}
    return tk_dict


def tk_plot(tf_dataframe, title, image_name='./images/contribution_rate.jpg'):
    import numpy as np
    import matplotlib.pyplot as mp

    data = tf_dataframe.values[0]
    mp.figure("tk_plot", facecolor='lightgray', figsize=(20, 10))
    mp.title(title)
    mp.grid(linestyle=':', axis='y')
    mp.xlabel('Features')
    mp.ylabel('Tk')
    x = np.arange(1, data.size + 1)
    mp.bar(
        x,
        data,
        color='dodgerblue',
        alpha=0.8
    )
    # 优化x轴刻度文本
    mp.xticks(x, list(tf_dataframe.columns), rotation=90)
    # mp.legend()
    # mp.show()
    mp.savefig(image_name, dpi=150, bbox_inches='tight', pad_inches=0.5)


def save_new_data(tk_dict):
    inputfile = None
    outputfile = None
    # 对贡献率排序
    sorted_tk_list = sorted(tk_dict.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_tk_list)
    pc = []  # 保存主成分特征
    dk = 0  # 保存累计贡献率
    # 保留主成分特征, 直到累计贡献率达到95%
    for item in sorted_tk_list:
        dk += item[1]
        pc.append(item[0])
        if dk > 0.95:
            break
    # print(pc)
    # print(len(pc))
    for class_ in ['class_1', 'class_2', 'class_3', 'class_4']:
        if class_ == 'class_1':
            inputfile = inputfile_1
            outputfile = outputfile_1
        elif class_ == 'class_2':
            inputfile = inputfile_2
            outputfile = outputfile_2
        elif class_ == 'class_3':
            inputfile = inputfile_3
            outputfile = outputfile_3
        elif class_ == 'class_4':
            inputfile = inputfile_4
            outputfile = outputfile_4
        data = pd.read_csv(inputfile)
        # 根据主成分重新构建训练数据(特征选择+标准化)
        new_samples = data.loc[:, pc]
        # 对特征数据标准化处理
        std_samples = sp.scale(new_samples)  # ndarray类型
        # ndarray 转 dataframe类型
        new_data = pd.DataFrame(std_samples, columns=pc)
        # 添加标签列
        new_data['label'] = data['label']
        new_data.to_csv(outputfile, index=False)


if __name__ == '__main__':
    # 计算贡献率
    tk_dict = get_pca_tk(inputfile_1)
    tf_dataframe = pd.DataFrame(tk_dict, index=['tk'])
    # 绘制贡献率直方图
    tk_plot(tf_dataframe, 'Contribution rate')
    # # 绘制贡献率top10直方图
    # tf_dataframe = tf_dataframe.iloc[:, :10]
    # tk_plot(tf_dataframe, 'Contribution rate Top10')
    # 保存只包含主成分的新训练数据(累计贡献度90%)
    save_new_data(tk_dict)


