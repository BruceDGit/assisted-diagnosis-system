"""
    对指标之间以及指标和疾病类别之间进行相关性分析
"""
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns

THRESHOLD = 0.7 # 阈值, 过滤相关系数
# tips = pd.read_csv('./data/class_4.dat')
CLASS_1_FILE = './data/class_1.dat'
CLASS_2_FILE = './data/class_2.dat'
CLASS_3_FILE = './data/class_3.dat'
CLASS_4_FILE = './data/class_4.dat'


class Plot:
    def __init__(self, threshold=0.7):
        self.threshold = threshold  # 阈值, 过滤相关系数
        self.data = pd.read_csv(CLASS_1_FILE)
        self.data_corr = self.data.corr()  # 相关性矩阵[皮尔逊]
        # 初始化时绘制相关性较强的指标的散点图
        self.subplot_scatter()

    def plot_corr_heatmap(self, class_=1):
        """
            绘制指标及标签互相之间的相关性热力图
        """
        # 读数据
        class_name = 'Class 1'
        img_name = './images/class_1_heapmap.jpg'
        if class_ == 1:
            pass
        elif class_ == 2:
            self.data = pd.read_csv(CLASS_2_FILE)
            class_name = 'Class 2'
            img_name = './images/class_2_heapmap.jpg'
        elif class_ == 3:
            self.data = pd.read_csv(CLASS_3_FILE)
            class_name = 'Class 3'
            img_name = './images/class_3_heapmap.jpg'
        elif class_ == 4:
            self.data = pd.read_csv(CLASS_4_FILE)
            class_name = 'Class 4'
            img_name = './images/class_4_heapmap.jpg'
        # 绘制热力图
        self.data_corr = self.data.corr()  # 相关性矩阵[皮尔逊]
        f, ax1 = mp.subplots(figsize=(14, 10))
        # 设置颜色
        # cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
        # sns.heatmap(tips_corr, cmap='RdBu', linewidths = 0.01, ax = ax1)
        ax = sns.heatmap(self.data_corr, linewidths=0.01, xticklabels=True, yticklabels=True, center=0.25)
        # 设置标题
        ax1.set_title(class_name, fontsize=18, position=(0.5, 1.05))
        # mp.show()
        f.savefig(img_name, dpi=150, bbox_inches='tight', pad_inches=0.5)

    def subplot_scatter(self):
        """
            提取相关系数绝对值大于0.7的指标进行单独绘图
        """
        # 提取满足条件的指标对
        res = []
        for idx in self.data_corr.index:
            for col in self.data_corr.columns:
                if abs(self.data_corr.loc[idx, col]) > THRESHOLD and self.data_corr.loc[idx, col] < 1 \
                        and (idx, col) not in res and (col, idx) not in res:
                    res.append((idx, col))
        # 绘制子图
        mp.figure('Subplot', facecolor='lightgray')
        subplot_width_ = 3
        subplot_height_ = len(res) // 3 + 1
        for i in range(1, len(res) + 1):
            mp.subplot(subplot_height_, subplot_width_, i)
            x = self.data[res[i - 1][0]]
            y = self.data[res[i - 1][1]]
            mp.title('PCCs=%.2f' % (self.data_corr.loc[res[i - 1][0], res[i - 1][1]]))
            mp.xlabel(res[i - 1][0])
            mp.ylabel(res[i - 1][1])
            mp.grid(linestyle=':')
            mp.scatter(x, y, s=6, alpha=0.5, label='')
        mp.tight_layout()
        # mp.show()
        mp.savefig('./images/corr_subplot.jpg', dpi=150, bbox_inches='tight', pad_inches=0.5)


if __name__ == '__main__':
    CLASS_1_FILE = '../data/class_1.dat'
    CLASS_2_FILE = '../data/class_2.dat'
    CLASS_3_FILE = '../data/class_3.dat'
    CLASS_4_FILE = '../data/class_4.dat'
    plot_obj = Plot()
    plot_obj.plot_corr_heatmap(1)
    plot_obj.plot_corr_heatmap(2)
    plot_obj.plot_corr_heatmap(3)
    plot_obj.plot_corr_heatmap(4)





