#coding=utf-8

import pandas as pd
import  numpy as np
import  xgboost as xgb
from  settings import  *
from  sklearn.ensemble import  GradientBoostingRegressor

user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')

#芝麻分的缺失值处理
#尝试通过树模型来预测缺失值，相关性较强的其他变量作为该模型特征
def zhimaScore_missingValue_process(df):
    numerical_df = df[numericalFeatures + ['user_id']]
    test_df = numerical_df[numerical_df['zhima_score'].isnull()]
    train_df = numerical_df[numerical_df['zhima_score'].notnull()]

    train_df.fillna(train_df.mean(), inplace = True)
    test_df.fillna(test_df.mean(), inplace = True)

    model_features = list(pd.read_excel(ROOT_DIR + 'featureEngineering/zhima_correlation_features.xlsx')['feature'])


    X = train_df[model_features]
    y = train_df['zhima_score']

    clf = GradientBoostingRegressor(loss='ls',
                                    alpha=0.9,
                                    n_estimators=500,
                                    learning_rate=0.05,
                                    max_depth=8,
                                    subsample=0.8,
                                    max_features=0.6,
                                    min_samples_split=9,
                                    max_leaf_nodes=10)
    print(X)
    print(y)
    model = clf.fit(X, y)

    test_X = test_df[model_features]
    result = model.predict(test_X)
    for score in result:
        print(score)


def job_level_missingValue(df):
    df['job_level'] = df['job_level'].fillna('其他')



#zhimaScore_missingValue_process()

