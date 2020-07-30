"""
    对指标之间以及指标和疾病类别之间进行相关性分析
"""
from models.Correlation import Plot

if __name__ == '__main__':
    plot_obj = Plot()
    plot_obj.plot_corr_heatmap(1)
    plot_obj.plot_corr_heatmap(2)
    plot_obj.plot_corr_heatmap(3)
    plot_obj.plot_corr_heatmap(4)






