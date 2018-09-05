#coding=utf-8

import  pandas as pd
import matplotlib.pyplot as plt

from  settings import  *
from  util.scorecard_functions import *


'''
badrate所有不单调的类别变量:
not_monotone_list = ['job_level', 'phone_province', 'phone_city', 'identity_province',
                    'identity_city',
                    'registerDate_hour', 'applyDate_hour']
'''

'''
1. 离散化程度高且无序的变量，直接用badrate进行编码
job_level
'''

not_monotone_list = []
train_data = pd.read_excel(ROOT_DIR + 'train.xlsx', encoding = 'utf-8')

def  visualization(var, x, y, x_label, y_label, title):
    plt.bar(x, y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(range(len(x) + 1), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/%s_badrate.png' %(var))
    plt.show()

for var in categoricalFeatures:
    if not BadRateMonotone(train_data, var, target=LABEL):
            not_monotone_list.append(var)

print(not_monotone_list)

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

var = 'applyDate_hour'
#monotone_analysis(train_data, var)

dicts, regroup = BinBadRate(train_data, var, LABEL)

print('dict:')
print(dicts)
print('regroup:')
print(regroup)

print(regroup[var].values)

visualization(var, regroup[var].values, list(regroup.bad_rate.values), x_label='hour', y_label='badrate', title='title')