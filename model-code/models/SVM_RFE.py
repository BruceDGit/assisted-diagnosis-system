import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


def rfe_processing(class_value = 1):
    if class_value == 1:
        input_file = './data/class_1.dat'
        output_file = './data/RFE_class_1.dat'
        img_file = './images/Cross_calidation_score_class_1.jpg'
    elif class_value == 2:
        input_file = './data/class_2.dat'
        output_file = './data/RFE_class_2.dat'
        img_file = './images/Cross_calidation_score_class_2.jpg'
    elif class_value == 3:
        input_file = './data/class_3.dat'
        output_file = './data/RFE_class_3.dat'
        img_file = './images/Cross_calidation_score_class_3.jpg'
    elif class_value == 4:
        input_file = './data/class_4.dat'
        output_file = './data/RFE_class_4.dat'
        img_file = './images/Cross_calidation_score_class_4.jpg'

    data = pd.read_csv(input_file)

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    # 标准化
    x_std = scale(x)

    # Create the RFE object and comput a cross-validated score.
    svc = SVC(kernel="linear")

    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='roc_auc')
    rfecv.fit(x_std, y)
    mask = rfecv.support_

    print("=> Optimal number of features: %d"%rfecv.n_features_)
    print("=> Length of the features list: %d"%len(rfecv.ranking_))
    print("=> Features importance ranking:\n %s"%rfecv.ranking_)
    print("=> feature masks:\n %s"%rfecv.support_)
    print("=> Model's cross calidation score: \n%s"%rfecv.grid_scores_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross calidation score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.tick_params(color='black', labelcolor='black', colors='black')
    # plt.show()
    plt.savefig(img_file, dpi=150, bbox_inches='tight', pad_inches=0.5)

    mask_all = np.append(mask, True)
    data_new = data.loc[:, mask_all]

    data_new.to_csv(output_file, index=False)
    print("=> Preview: \n%s"%pd.read_csv(output_file).head(5))
