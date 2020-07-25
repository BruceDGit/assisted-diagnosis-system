"""
    贡献率的另一种计算方法
"""
import pandas as pd
import numpy as np
import sklearn.preprocessing as sp

inputfile = "./data/class_4.dat"
outputfile = "./data/new_class_4.dat"

data = pd.read_csv(inputfile)

# 提取特征名
features = data.columns
features = list(features[:-1])

# 对特征数据标准化处理
std_samples = sp.scale(data.iloc[:, :-1])  # ndarray类型
# ndarray 转 dataframe类型
std_samples = pd.DataFrame(std_samples)

# 计算相关系数矩阵
corr_samples = std_samples.corr()

# 计算特征值和特征向量
res = np.linalg.eig(corr_samples)
# 保存特征值
eigenvalues = res[0]

# 计算贡献率, 得到贡献率字典
tk_dict = {feature: round(item / eigenvalues.sum(), 4) for item, feature in zip(eigenvalues, features)}
# 对贡献率排序
sorted_tk_list = sorted(tk_dict.items(), key=lambda item: item[1], reverse=True)

pc = []  # 保存主成分特征
dk = 0    # 保存累计贡献率
# 保留主成分特征, 直到累计贡献率达到95%
for item in sorted_tk_list:
    dk += item[1]
    pc.append(item[0])
    if dk > 0.95:
        break

# 根据主成分重新构建训练数据(特征选择+标准化)
new_samples = data.loc[:, pc]
# 对特征数据标准化处理
std_samples = sp.scale(new_samples)  # ndarray类型
# ndarray 转 dataframe类型
new_data = pd.DataFrame(std_samples, columns=pc)

# 添加标签列
new_data['label'] = data['label']
# print(new_data)

new_data.to_csv(outputfile, index=False)
