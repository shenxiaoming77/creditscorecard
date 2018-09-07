#coding=utf-8

import  pandas as pd
import  numpy as np


from  settings import  *

user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')
user_info_df[categoricalFeatures].fillna('null', inplace = True)
user_info_df[numericalFeatures].fillna(-1, inplace = True)

#分析芝麻分与其他连续变量的相关性，根据强相关特征来进行用户分群，从而对同组用户进行芝麻分的缺失填补
def zhima_score_analysis():

    numerical_data = user_info_df[numericalFeatures]

    x = 'zhima_score'

    x_data = numerical_data[x].astype('int')
    '''
    zhima_score与auth_level, company_college_length 这两个连续变量关联度较高
    '''
    for y in numericalFeatures:
        print(y)
        if x != y:
            y_data = numerical_data[y].astype('int')
            print(x, '      ', y, '     ', np.corrcoef(x_data, y_data)[0, 1])


def phone_identity_province_analysis():
    phone_province_groupby = user_info_df.groupby('phone_province')[LABEL]
    groupby_count = phone_province_groupby.count()
    groupby_sum = phone_province_groupby.sum()

    count_list = list(groupby_count)
    sum_list = list(groupby_sum)
    keys = list(dict(list(phone_province_groupby)).keys())

    for i in range(len(keys)):
        pri


phone_identity_province_analysis()