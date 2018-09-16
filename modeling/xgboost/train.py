#coding=utf-8
import  pandas as pd
import  numpy as np
import  xgboost as xgb
import os,random

from  settings import  *
from  util.featureEngineering_functions import *

class XGBClassifier():
    def __init__(self):
        self.train_df = pd.read_excel(ROOT_DIR + 'train.xlsx', encoding='utf-8')


    def model_cv(self, feature_file):
        print(self.train_df.columns)
        features = loadFeatures(feature_file)
        print(features)
        X = self.train_df[features]
        y = self.train_df[LABEL]

        dtrain = xgb.DMatrix(X, label = y)

        #通过cv找最佳的nround
        cv_log = xgb.cv(xgb_params,
                        dtrain,
                        num_boost_round=  5000,
                        nfold=3,
                        metrics='auc',
                        early_stopping_rounds=200,
                        seed = 2018)
        print(cv_log)
        best_auc= cv_log['test-auc-mean'].max()

        cv_log['nb'] = cv_log.index
        cv_log.index = cv_log['test-auc-mean']

        print('cv log nb dict:')
        print(cv_log.nb.to_dict())
        nround = cv_log.nb.to_dict()[best_auc]

        print(nround)

    def model_train(self, feature_file):
        features = loadFeatures(feature_file)

        X = self.train_df[features]
        y = self.train_df[LABEL]

        dtrain = xgb.DMatrix(X, label = y)


if __name__ == '__main__':
    clf = XGBClassifier()
    clf.model_cv('features')