#coding=utf-8

import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  pickle

from  settings import  *

user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')


#所有原始特征的空值情况统计
def  null_counts_analysis():
    null_count_dict = {}
    total = len(user_info_df['job_level'])
    features_list = numericalFeatures + categoricalFeatures

    for var in features_list:
        null_count = len(user_info_df[var]) - user_info_df[var].count()
        null_count_dict[var] = null_count

    #升序排列
    sorted_dict = sorted(null_count_dict.items(), key = lambda  d : d[1], reverse = True)
    x = []
    y = []
    i = 0
    for item in sorted_dict:
        if i > 5:
            break
        x.append(item[0])
        y.append(item[1] * 1.0 / total)
        i += 1

    x_label='features'
    y_label= 'NAN Count Num'
    title='特征空值数量统计'
    plt.bar(x, y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(range(len(x) + 1), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/null_count.png')
    plt.show()

#分析芝麻分与其他连续变量的相关性，根据强相关特征来进行用户分群，从而对同组用户进行芝麻分的缺失填补
def zhima_score_analysis():

    df = user_info_df[numericalFeatures].fillna(-1)
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


def phone_province_analysis():
    df = user_info_df[categoricalFeatures].fillna('null', inplace = True)
    phone_province_groupby = df.groupby('phone_province')[LABEL]
    groupby_count = phone_province_groupby.count()
    groupby_sum = phone_province_groupby.sum()

    badrates = []

    count_list = list(groupby_count)
    sum_list = list(groupby_sum)
    keys = list(dict(list(phone_province_groupby)).keys())

    badrate_dict = {}
    for i in range(len(keys)):
        badrate_dict[keys[i]] = (sum_list[i] * 1.0 / count_list[i])
        print(keys[i], '    ', sum_list[i], '   ', count_list[i])

    #升序排列
    sorted_dict = sorted(badrate_dict.items(), key = lambda  d : d[1], reverse = True)

    with open(ROOT_DIR + 'phone_province_badrate.pkl', 'wb') as file:
        pickle.dump(sorted_dict, file)

    x = []
    y = []
    for item in sorted_dict:
        x.append(item[0])
        y.append(item[1])
    x_label='phone province'
    y_label= 'badrate'
    title='手机号所在省份的整体坏账率'
    plt.bar(x, y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(range(len(x) + 1), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/phone_province_badrate.png')
    plt.show()

def identity_province_analysis():
    df = user_info_df[categoricalFeatures].fillna('null')
    identity_province_groupby = df.groupby('identity_province')[LABEL]
    groupby_count = identity_province_groupby.count()
    groupby_sum = identity_province_groupby.sum()

    badrates = []

    count_list = list(groupby_count)
    sum_list = list(groupby_sum)
    keys = list(dict(list(identity_province_groupby)).keys())

    badrate_dict = {}
    for i in range(len(keys)):
        badrate_dict[keys[i]] = (sum_list[i] * 1.0 / count_list[i])
        print(keys[i], '    ', sum_list[i], '   ', count_list[i])

    #升序排列
    sorted_dict = sorted(badrate_dict.items(), key = lambda  d : d[1], reverse = False)

    with open(ROOT_DIR + 'identity_province_badrate.pkl', 'wb') as file:
        pickle.dump(sorted_dict, file)

    x = []
    y = []
    for item in sorted_dict:
        x.append(item[0])
        y.append(item[1])
    x_label='identity province'
    y_label= 'badrate'
    title='户籍所在省份的整体坏账率'
    plt.bar(x, y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(range(len(x) + 1), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/identity_province_badrate.png')
    plt.show()

def phone_city_analysis():
    phone_city_df = user_info_df['phone_city'].fillna('null')
    phone_city_df['phone_city_bin'] = phone_city_df.apply()

null_counts_analysis()
phone_province_analysis()
identity_province_analysis()