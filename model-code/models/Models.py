import sklearn.linear_model as lm
import sklearn.ensemble as se
import sklearn.svm as svm
import sklearn.metrics as sm

from models.DataLoader import load_data


class ClassifierModels:
    def __init__(self, train_x, test_x, train_y, test_y):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

    def random_forest_classifier(self):
        """
            随机森林分类器
            max_depth：   决策树最大深度10
            n_estimators：决策树数目
            random_state: 随机种子(保证每次执行使用的同一批打乱的数据集)
            class_weight: 'balanced' 样本均衡
        """
        model = se.RandomForestClassifier(max_depth=10, n_estimators=1000, random_state=7, class_weight='balanced')
        model.fit(self.train_x, self.train_y)
        pred_test_y = model.predict(self.test_x)
        print(sm.classification_report(self.test_y, pred_test_y))

    def lr_classifier(self):
        """
            逻辑回归分类器
            solver: 'liblinear' 迭代优化算法
            C: 正则化
        """
        model = lm.LogisticRegression(solver='liblinear', C=10, class_weight='balanced', random_state=7)
        model.fit(self.train_x, self.train_y)
        pred_test_y = model.predict(self.test_x)
        print(sm.classification_report(self.test_y, pred_test_y))

    def ridge_classifier(self):
        """
            岭回归分类器
            normalize: 标准化处理
        """
        model = lm.RidgeClassifier(normalize=True, class_weight='balanced', random_state=7)
        model.fit(self.train_x, self.train_y)
        pred_test_y = model.predict(self.test_x)
        print(sm.classification_report(self.test_y, pred_test_y))

    def svm_linear_classifier(self):
        """
            SVM - 线性核函数
        """
        model = svm.SVC(kernel='linear', class_weight='balanced', random_state=7)
        model.fit(self.train_x, self.train_y)
        pred_test_y = model.predict(self.test_x)
        print(sm.classification_report(self.test_y, pred_test_y))

    def svm_poly_classifier(self):
        """
            SVM - 多项式核函数
        """
        model = svm.SVC(kernel='poly', degree=3, class_weight='balanced', random_state=7)
        model.fit(self.train_x, self.train_y)
        pred_test_y = model.predict(self.test_x)
        print(sm.classification_report(self.test_y, pred_test_y))

    def svm_rbf_classifier(self):
        """
            SVM - 径向基核函数
        """
        model = svm.SVC(kernel='rbf', C=10, class_weight='balanced', random_state=7)
        model.fit(self.train_x, self.train_y)
        pred_test_y = model.predict(self.test_x)
        print(sm.classification_report(self.test_y, pred_test_y))


def class_1(file_path, data_type='normal'):
    train_x, test_x, train_y, test_y = load_data(file_path)
    models = ClassifierModels(train_x, test_x, train_y, test_y)
    if data_type == 'normal':
        models.svm_linear_classifier()
    elif data_type == 'RFE':
        models.svm_rbf_classifier()


def class_2(file_path, data_type='normal'):
    train_x, test_x, train_y, test_y = load_data(file_path)
    models = ClassifierModels(train_x, test_x, train_y, test_y)
    if data_type == 'normal':
        models.svm_poly_classifier()
        # models.svm_linear_classifier()
    elif data_type == 'RFE':
        models.svm_rbf_classifier()


def class_3(file_path, data_type='normal'):
    train_x, test_x, train_y, test_y = load_data(file_path)
    models = ClassifierModels(train_x, test_x, train_y, test_y)
    if data_type == 'normal':
        models.random_forest_classifier()
    elif data_type == 'RFE':
        models.svm_linear_classifier()


def class_4(file_path, data_type='normal'):
    train_x, test_x, train_y, test_y = load_data(file_path)
    models = ClassifierModels(train_x, test_x, train_y, test_y)
    if data_type == 'normal':
        models.random_forest_classifier()
    elif data_type == 'RFE':
        models.svm_poly_classifier()


def classifier_test(file_path):
    train_x, test_x, train_y, test_y = load_data(file_path)
    models = ClassifierModels(train_x, test_x, train_y, test_y)
    print("\nrandom_forest_classifier: ")
    models.random_forest_classifier()
    print("\nlr_classifier: ")
    models.lr_classifier()
    print("\nridge_classifier: ")
    models.ridge_classifier()
    print("\nsvm_linear_classifier: ")
    models.svm_linear_classifier()
    print("\nsvm_poly_classifier: ")
    models.svm_poly_classifier()
    print("\nsvm_rbf_classifier: ")
    models.svm_rbf_classifier()