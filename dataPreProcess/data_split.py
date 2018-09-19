#coding = utf-8

import  pandas as pd
from sklearn.model_selection import  train_test_split
from  settings import *


def train_test_split(data, features_list):
    label = 'loan_status'
    X = data[features_list]
    y = data[label]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True, stratify=y)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    '''
    train_sum = train_y.sum()
    test_sum = test_y.sum()

    print(train_sum)
    print(test_sum)

    print(train_sum / 12474.0)
    print(test_sum / 5347.0)
    '''

    train = pd.concat([x_train, y_train], axis = 1)
    test = pd.concat([x_test, y_test], axis = 1)


    #train.to_excel(ROOT_DIR + 'train.xlsx',index= None, encoding='utf-8')
    #test.to_excel(ROOT_DIR + 'test.xlsx', index= None, encoding='utf-8')
    return  (x_train, x_test, y_train, y_test)