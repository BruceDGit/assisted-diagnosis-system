from models.SVM_RFE import rfe_processing


if __name__ == '__main__':
    print('*'*40, 'class 1', '*'*40)
    rfe_processing(1)
    print('*'*40, 'class 2', '*'*40)
    rfe_processing(2)
    print('*'*40, 'class 3', '*'*40)
    rfe_processing(3)
    print('*'*40, 'class 4', '*'*40)
    rfe_processing(4)