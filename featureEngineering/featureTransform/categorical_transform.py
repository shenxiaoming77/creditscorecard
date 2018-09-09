#coding=utf-8
import  pandas as pd
import  numpy as np

from  settings import  *
from  util.featureEngineering_functions import *

train = pd.read_excel(ROOT_DIR + 'train.xlsx', encoding = 'utf-8')

transformed_feature_list = []


def br_score_transform():
    br_df = train['br_score'].apply(assign_br_score_bin)

    print(br_df)
    train['br_score_bin'] = br_df
    transformed_feature_list.append('br_score')

def zhima_score_transform():
    zhima_score_df = train['zhima_score']

    feature_list = numericalFeatures + categoricalFeatures

    user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')

    numerical_data = user_info_df[numericalFeatures]
    numerical_data.fillna(-1, inplace = True)






def save():
    train.to_excel(FE_DIR + 'train_transformed.xlsx', index = None)

    with open(FE_DIR + 'transformed_feature_list.txt') as f:
        f.write(transformed_feature_list)
        f.close()



#br_score_transform()
zhima_score_transform()