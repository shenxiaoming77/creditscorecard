#coding=utf-8

import pandas as pd
import  numpy as np
import  xgboost as xgb
from  settings import  *
from  sklearn.ensemble import  GradientBoostingRegressor
from  dataPreProcess.data_split import *

#存在空值情况的原始特征，需要进行空值插入
'''
job_level
phone_province
phone_city
identity_province
identity_city
zhima_score
network_len
occupation
company_college_length
br_score
'''


#芝麻分的缺失值处理
#尝试通过树模型来预测缺失值，相关性较强的其他变量作为该模型特征
def zhimaScore_missingValue_ByModel(df):
    numerical_df = df[FEATURE_DICT['numericalFeatures'] + ['user_id']]
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

#通过相似用户群聚类，按群内的均值进行空值补足
def zhimaScore_missingValue_ByCluster(df, zhimaScore_df):
    zhimaScore = list(zhimaScore_df['zhima_score'])
    df['zhima_score_tmp'] = zhimaScore
    df.drop('zhima_score', axis = 1, inplace = True)
    df.rename(columns={'zhima_score_tmp':'zhima_score'}, inplace = True)

#job_level缺失值补足
def job_level_missingValue(df):
    df['job_level'] = df['job_level'].fillna('missing')

#phone_city缺失值补足
def phone_city_missingValue(df):
    df['phone_city'] = df['phone_city'].fillna('missing')

#phone_province缺失值
def phone_province_missingValue(df):
    df['phone_province'] = df['phone_province'].fillna('missing')

#identity_city缺失值
def identity_city_missingValue(df):
    df['identity_city'] = df['identity_city'].fillna('missing')

#identity_province缺失值
def identity_province_missingValue(df):
    df['identity_province'] = df['identity_province'].fillna('missing')

#br_score缺失值
def br_score_missingValue(df):
    df['br_score'] = df['br_score'].fillna(-1)

#occupation缺失值
def occupation_missingValue(df):
    df['occupation'] = df['occupation'].fillna('missing')


#network_len缺失值
def network_len_missingValue(df):
    df['network_len'] = df['network_len'].fillna('missing')

#company_college_length缺失值
def company_college_length_missingValue(df):
    df['company_college_length'] = df['company_college_length'].fillna(df['company_college_length'].mean())


