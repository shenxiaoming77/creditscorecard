#coding=utf-8

import  pandas as pd
import matplotlib.pyplot as plt

from  settings import  *
from  util.scorecard_functions import *

categoricalFeatures = FEATURE_DICT['categoricalFeatures']

not_monotone_list = []
train_data = pd.read_excel(ROOT_DIR + 'transformed_train.xlsx', encoding = 'utf-8')

def  visualization(var, x, y, x_label, y_label, title):
    plt.bar(x, y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(range(len(x) + 1), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/%s_badrate.png' %(var))
    plt.show()



def monotone_analysis(train_data, var):
    #print(train_data.groupby(var)[LABEL].mean())
    #print(train_data.groupby(var)[LABEL].count())
    mean_dict = {}
    count_dict = {}
    group_data = train_data.groupby(var)[LABEL]
    mean_data = group_data.mean()
    count_data = group_data.count()
    keys = list(dict(list(group_data)).keys())
    values = list(mean_data)
    counts = list(count_data)


    for i in range(len(keys)):
        key = keys[i]
        value = values[i]
        mean_dict[key] = value
        count_dict[key] = counts[i]

    sorted_dict = sorted(mean_dict.items(), key = lambda  d : d[1], reverse=True)


    for item in sorted_dict:
        key = item[0]
        print(item[0], '    ', item[1], '   ', count_dict[key])



#针对离散化程度高且无序的变量，需要观察每个bin的badrate分布情况，对于极端分布
#job_level, phone_province两个特征存在零坏样本的bin
def badrate0_analysis(df, col, target, direction):
    print(col)
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B

    result = regroup[direction].map(lambda x: (str(x == 0)))
    print(result)
    if 'True' in set(result):
        print(direction + 'rate0 is true')




'''
1. 离散化程度高且无序的变量，直接用badrate进行编码，无需进一步的分箱合并，保证badrate单调性
job_level, phone_province,identity_province,occupation

2. badrate不单调并且需要卡方分箱合并的类别变量:
'auth_level': == 0, == 1, == 2, >= 3
'network_len': ==0, ==1, >=2
'identity_city_classification': ==0, ==1, >=2 and <= 3, ==4, >=5
'phone_city_classification':==0, >= 1 and <=2, ==3, ==4, >=5
'br_score_classification':==0, >=1 and <= 2, ==3
'user_age_classification':==0, ==1, >=2
'''

#无序且高离散化变量，不需要进一步分箱合并，直接基于每个bin的badrate进行编码
unordered_categorical_variable = ['job_level', 'phone_province','identity_province','occupation']

#获取所有badrate不单调的类别变量，逐一分析是否需要进行卡方分箱合并
for var in categoricalFeatures:
    print(var)
    if not BadRateMonotone(train_data, var, target=LABEL):
        if var not in unordered_categorical_variable:
            not_monotone_list.append(var)
print(not_monotone_list)


# var = 'user_age_classification'
# #monotone_analysis(train_data, var)
#
# print(set(train_data[var]))
#
# dicts, regroup = BinBadRate(train_data, var, LABEL)
#
# print('dict:')
# print(dicts)
# print('regroup:')
# print(regroup)
#
# print(regroup[var].values)
# visualization(var, regroup[var].values, list(regroup.bad_rate.values), x_label= var, y_label='badrate', title='title')

# target = 'loan_status'
# #features = ['job_level', 'phone_province','identity_province','occupation']
# features = ['zhima_score_classification']
# direction = 'bad'
# for var in features:
#     badrate0_analysis(train_data, var, target, direction)
#
# result = MergeBad0(train_data, 'zhima_score_classification', LABEL, direction)
# print(result)
#
# train_data['zhima_score_classification'] = train_data['zhima_score_classification'].map(lambda x : result[x])
# print(train_data['zhima_score_classification'])
