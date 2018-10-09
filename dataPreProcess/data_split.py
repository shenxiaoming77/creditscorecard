#coding = utf-8

import  pandas as pd
from sklearn.model_selection import  train_test_split
from  settings import *


def train_test_split_func(data, features_list, size):
    label = 'loan_status'
    X = data[features_list]
    y = data[label]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= size, shuffle=True, stratify = y)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    return  (x_train, x_test, y_train, y_test)