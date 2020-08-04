"""
    计算贡献率
    保存只包含主成分的新训练数据(累计贡献度95%)
"""
from models.PCA import *


if __name__ == '__main__':
    # 计算贡献率
    tk_dict = get_pca_tk(inputfile_1)
    tf_dataframe = pd.DataFrame(tk_dict, index=['tk'])
    # 绘制贡献率直方图
    tk_plot(tf_dataframe, 'Contribution rate')
    # # 绘制贡献率top10直方图
    # tf_dataframe = tf_dataframe.iloc[:, :10]
    # tk_plot(tf_dataframe, 'Contribution rate Top10')
    # 保存只包含主成分的新训练数据(累计贡献度95%)
    save_new_data(tk_dict)





