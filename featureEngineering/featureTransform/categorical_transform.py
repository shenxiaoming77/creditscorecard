#coding=utf-8
import  pandas as pd
import  numpy as np

from  settings import  *

from datetime import datetime,date

dayOfWeek = datetime.now().weekday()
print (dayOfWeek)

dayOfWeek = datetime.today().weekday()
print (dayOfWeek)

train = pd.read_excel(ROOT_DIR + 'train.xlsx', encoding = 'utf-8')

transformed_feature_list = []

def assign_br_score_bin(x):

    print(x)
    if str(x).find("null") >= 0:
        print('is null')
        return  0
    score = int(x)
    if score < 500:
        return 1
    elif score < 550 & score >= 500:
        return  2
    elif score < 600 & score >= 550:
        return  3
    elif score < 650 & score >= 600:
        return  4
    else:
        return  5


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