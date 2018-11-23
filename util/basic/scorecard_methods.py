
# -*- coding: utf-8 -*-

from util.basic.scorecard_functions import *
import random
import numpy as np
import pandas as pd

import math
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
import json



def is_overdue(x):
    '''1-逾期, 0-未逾期, -1-未到期'''
    if  x == '\\N':
        return -1
    elif x <= 0:
        return 0
    else:
        return 1

def assign_group(x, bin):
    N = len(bin)
    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return max(bin)+100
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]

def assign_bin(x, cut_point, special_attr = []):
    num_Bins = len(cut_point) + 1
    # print(x)
    if x in special_attr:
        i = special_attr.index(x) + 1
        return 'Bin {}'.format(0-i)
    if x <= cut_point[0]:
        return 'Bin 0'
    elif x > cut_point[-1]:
        return 'Bin {}'.format(num_Bins-1)
    else:
        for i in range(0, num_Bins-1):
            if cut_point[i] < x <= cut_point[i+1]:
                return 'Bin {}'.format(i+1)


def split_data(df, col, num_bins=5):
    '''划分数据'''
    df_temp = df.copy()
    # 数据集数量
    N_dataSet = df_temp.shape[0]
    #每箱数据个数
    n_bin = N_dataSet // num_bins
    #切分点编号
    split_point_index = [i*n_bin for i in range(1, num_bins)]
    #对数据进行排序
    sorted_values = sorted(df_temp[col].values)
    #获得切分点
    split_point = [sorted_values[index] for index in split_point_index ]
    split_point = sorted(list(set(split_point)))

    return(split_point)

def binning(df, col, cut_points, labels=None):
    # Define min and max values:
    minval = df[col].min()
    maxval = df[col].max()
    # 利用最大值和最小值创建分箱点的列表
    break_points = [minval] + cut_points + [maxval]
    # 如果没有标签，则使用默认标签0 ... (n-1)
    if not labels:
        labels = range(len(cut_points) + 1)
    # 使用pandas的cut功能分箱
    colBin = pd.cut(df[col], bins=break_points, labels=labels, include_lowest=True)
    return colBin

def get_labels(df, col, cut_points):
    minval, maxval = min(df[col]), max(df[col])
    break_points = [minval] + cut_points + [maxval]
    labels = []
    labels.append('[%d, %d]' % (break_points[0], break_points[1]))
    for i in range(1, len(break_points) - 1):
        labels.append('(%d, %d]' % (break_points[i], break_points[i + 1]))
    return labels

def BadRate(df, col, target):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :return: 每箱的坏样本率，以及总体的坏样本率
    '''
    total = df[col].value_counts().reset_index().rename(columns={col: 'total', 'index': col})
    bad = df.groupby(col)[target].sum().reset_index().rename(columns={target: 'bad'})
    regroup = pd.merge(total, bad, on=col)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    regroup.sort_values(by='bad_rate', inplace=True)
    return regroup

def cutoff_bin(user_df, col, target, minBinPcnt=0):
    var_bin_list = []
    continous_merged_dict = {}
    # print("{} is in processing".format(col))
    if -1 not in set(user_df[col]):   #－1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
        max_interval = 5   #分箱后的最多的箱数
        cutOff = ChiMerge(user_df, col, target, max_interval=max_interval,special_attribute=[],minBinPcnt=minBinPcnt)
        user_df[col+'_Bin'] = user_df[col].map(lambda x: AssignBin(x, cutOff,special_attribute=[]))
        monotone = BadRateMonotone(user_df, col+'_Bin', target)   # 检验分箱后的单调性是否满足
        while(not monotone):
            # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
            max_interval -= 1
            cutOff = ChiMerge(user_df, col, target, max_interval=max_interval, special_attribute=[], minBinPcnt=minBinPcnt)
            user_df[col + '_Bin'] = user_df[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
            if max_interval == 2:
                # 当分箱数为2时，必然单调
                break
            monotone = BadRateMonotone(user_df, col + '_Bin', target)
        newVar = col + '_Bin'
        user_df[newVar] = user_df[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
        var_bin_list.append(newVar)
    else:
        print('有特殊值-1')
        max_interval = 5
        # 如果有－1，则除去－1后，其他取值参与分箱
        cutOff = ChiMerge(user_df, col, target, max_interval=max_interval, special_attribute=[-1],
                                      minBinPcnt=minBinPcnt)
        print('chi merge first:')
        print(cutOff)
        user_df[col + '_Bin'] = user_df[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
        monotone = BadRateMonotone(user_df, col + '_Bin', target)
        while (not monotone):
            max_interval -= 1
            print('max_interval = ', max_interval)
            # 如果有－1，－1的bad rate不参与单调性检验
            cutOff = ChiMerge(user_df, col, target, max_interval=max_interval, special_attribute=[-1], minBinPcnt=minBinPcnt)
            user_df[col + '_Bin'] = user_df[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
            if max_interval == 3:
                # 当分箱数为2时，必然单调
                break
            # print(max_interval)
            monotone = BadRateMonotone(user_df, col + '_Bin', target)
        newVar = col + '_Bin'
        print('last max_interval:')
        print(max_interval)
        print('cutOff:')
        print(cutOff)
        user_df[newVar] = user_df[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
        var_bin_list.append(newVar)
    continous_merged_dict[col] = cutOff
    return continous_merged_dict

def generate_bad_rate(df, col, target, num_bins=5):
    df_temp = df.copy()
    cut_points = split_data(df_temp, col, num_bins=num_bins)
    labels = get_labels(df_temp, col, cut_points)
    new_col = col + 'lv'
    df_temp[new_col] = binning(df_temp, col, cut_points, labels=labels)
    regroup = BadRate(df_temp, new_col, target)
    return(regroup)

def equal_width_split_data(df, col, num_bins=5):
    '''等宽切分数据'''
    df_temp = df.copy()
    minval, maxval = df_temp[col].min(), df_temp[col].max() #获得最小值和最大值
    n_bin = (maxval - minval) * 1.0 // (num_bins) #每箱间隔
    split_points = [minval + i*n_bin for i in range(1, num_bins)] #获得切分点
    split_points = sorted(split_points)
    return split_points

def equal_frequency_split_data(df, col, num_bins=5):
    '''等频切分数据'''
    df_temp = df.copy()
    N_dataSet = df_temp.shape[0] # 数据集数量
    n_bin = N_dataSet * 1.0 // num_bins # 每箱数据个数
    split_point_index = [i * n_bin for i in range(1, num_bins)] # 切分点编号
    sorted_values = sorted(df_temp[col].values) # 对数据进行排序
    split_points = [sorted_values[index] for index in split_point_index] # 获得切分点
    split_points = sorted(list(set(split_points)))
    return split_points

def equal_width_binning(df, col, target, num_bins=5, special_attr=None):
    '''
    等宽分箱
    :param df: 数据框
    :param col: 列名，也就是所要处理的字段
    :param target: 标签
    :param num_bins: 分箱数
    :param special_attr: 特殊的值，如用-1代表空值
    :return: 逾期分布情况
    '''
    #######################划分数据#####################
    df_temp = df.copy()
    if special_attr is not None:
        df_temp = df_temp.loc[~df_temp[col].isin(special_attr)]
        num_bins -= 1
    minval, maxval = df_temp[col].min(), df_temp[col].max()
    n_bin = (maxval-minval) * 1.0 // (num_bins)
    split_points = [minval, maxval]

    for i in range(1, num_bins):
        split_points.append(minval + i*n_bin)
    break_points = sorted(split_points)
    # print(break_points)
    #######################生成标签######################
    labels = []
    labels.append('[{}, {}]'.format(break_points[0], break_points[1]))
    for i in range(1, len(break_points) - 1):
        labels.append('({}, {}]'.format(break_points[i], break_points[i + 1]))
    ########################分箱#########################
    # print(df_temp[col])
    df_temp[col] = pd.cut(df_temp[col], bins=break_points, labels=labels, include_lowest=True)
    # df_temp[col] = df_temp[col].apply(lambda x: assign_bin(x, break_points, special_attr=[-1]))
    # print(df_temp[col])
    #######################计算逾期率###########################
    regroup = BadRate(df_temp, col, target)
    if special_attr is not None:
        regroup_special = BadRate(df.loc[df[col].isin(special_attr)], col, target)
        regroup = pd.concat([regroup_special, regroup], ignore_index=True)
    return regroup

def equal_frequency_binning(df, col, target, num_bins=5, special_attr=None):
    '''
    等频分箱
    :param df: 数据框
    :param col: 列名，也就是所要处理的字段
    :param target: 标签
    :param num_bins: 分箱数
    :param special_attr: 特殊的值，如用-1代表空值
    :return: 逾期分布情况
    '''
    #######################划分数据######################
    #df[col] = df[col].apply(lambda x: '[%d]' % x if x in special_attr else x)
    df_temp = df.copy()
    if special_attr is not None:
        df_temp = df_temp.loc[~df_temp[col].isin(special_attr)]
        num_bins -= 1
    ######################等频切分#######################
    # 数据集数量
    N_dataSet = df_temp.shape[0]
    woe_dict = {}
    for bins in range(2, num_bins+1):
        n_bin = N_dataSet // bins
        split_point_index = [i * n_bin for i in range(1, bins)]
        sorted_values = sorted(df_temp[col].values)
        split_point = [sorted_values[index] for index in split_point_index]
        split_point = sorted(list(set(split_point)))
        if special_attr is not None:
            df_temp[col + '_Bin'] = df_temp[col].apply(lambda x: assign_bin(x, split_point, special_attr=[-1]))
        else:
            df_temp[col + '_Bin'] = df_temp[col].apply(lambda x: assign_bin(x, split_point))
        woe = sum(woe_iv(df_temp, col+'_Bin', target)['woe'].values())
        # print(bins, woe)
        woe_dict[woe] = split_point

    split_point = woe_dict[max(woe_dict.keys())]
    # #每箱数据个数
    # n_bin = N_dataSet // (num_bins)
    # #切分点编号
    # split_point_index = [i*n_bin for i in range(1, num_bins)]
    # #对数据进行排序
    # sorted_values = sorted(df_temp[col].values)
    # #获得切分点
    # split_point = [sorted_values[index] for index in split_point_index ]
    # split_point = sorted(list(set(split_point)))
    # if special_attr is not None:
    #     df_temp[col + '_Bin'] = df_temp['user_age'].apply(lambda x: assign_bin(x, split_point))
    # else:
    #     df_temp[col + '_Bin'] = df_temp['user_age'].apply(lambda x: assign_bin(x, split_point, special_attr=[-1]))
    # print(woe_iv(df_temp, col+'_Bin', target)['iv'])
    #######################生成标签######################
    minval, maxval = min(df_temp[col]), max(df_temp[col])
    break_points = [minval] + split_point + [maxval]
    break_points = sorted(list(set(break_points)))
    labels = []
    labels.append('[{}, {}]'.format(break_points[0], break_points[1]))
    for i in range(1, len(break_points) - 1):
        labels.append('({}, {}]'.format(break_points[i], break_points[i + 1]))

    ########################分箱#########################
    # print(break_points)
    df_temp[col] = pd.cut(df_temp[col], bins=break_points, labels=labels, include_lowest=True)
    #######################计算逾期率###########################
    regroup = BadRate(df_temp, col, target)
    if special_attr is not None:
        regroup_special = BadRate(df.loc[df[col].isin(special_attr)], col, target)
        regroup = pd.concat([regroup_special, regroup], ignore_index=True)
    return [regroup, split_point]

# 计算KS值
def KS(df, col, target):
    total = df.groupby([col])[target].count()
    bad = df.groupby([col])[target].sum()
    regroup = pd.DataFrame({'total': total, 'bad': bad})

    regroup['good'] = regroup['total'] - regroup['bad']
    regroup[col] = regroup.index
    regroup = regroup.sort_values(by=col, ascending=True)
    regroup.index = range(len(regroup))
    regroup['badCumRate'] = regroup['bad'].cumsum() / regroup['bad'].sum()
    regroup['goodCumRate'] = regroup['good'].cumsum() / regroup['good'].sum()
    regroup['totalCumRate'] = regroup['total'].cumsum() / regroup['total'].sum()
    regroup['ks'] = regroup.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return regroup

def minAbsValue(lst):
    min_value = lst[0]
    min_abs = abs(lst[0])
    for elem in lst:
        if abs(elem) < min_abs:
            min_abs = abs(elem)
            min_value = elem
    return min_value

def split_dataset(df, col, target):
    regroup_ks = KS(df, col, target)
    max_ks = regroup_ks['ks'].max()
    min_ks = regroup_ks['ks'].min()
    ks = max_ks if abs(max_ks) >= abs(min_ks) else min_ks
    cutoff_value = regroup_ks[regroup_ks['ks'] == ks][col].values[0]
    df1 = df[df[col] > cutoff_value][[col, target]]
    df2 = df[df[col] <= cutoff_value][[col, target]]
    return {cutoff_value: [df1, df2]}

def getCutPoint(df, col, target, max_bins=5, special_attr=None, minBinPcnt=0):
    '''根据ks分箱，未完待续……'''
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_bins:
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return (colLevels)
    else:
        cutoffpoints = []
        if special_attr is not None:
            df_clean = df.loc[~df[col].isin(special_attr)]
            max_points = max_bins - 1 - len(special_attr)
        else:
            df_clean = df.copy()
            max_points = max_bins - 1
        df_list = []
        df_list.append(df_clean)
        N = df_clean.shape[0]
        while len(cutoffpoints) < max_points:
            df_temp = []
            for df_t in df_list:
                result = split_dataset(df_t, col, target)
                cutoff_value = list(result.keys())[0]
                df1, df2 = result[cutoff_value]
                # 每箱全为好或坏样本的合并
                df1.index = range(df1.shape[0])
                df2.index = range(df2.shape[0])
                df1_ks = KS(df1, col, target)
                df2_ks = KS(df2, col, target)
                badRate1 = df1_ks['bad'].sum() * 1.0 / df1_ks['total'].sum()
                badRate2 = df2_ks['bad'].sum() * 1.0 / df2_ks['total'].sum()
                if badRate1 in [0, 1] or badRate2 in [0, 1]:
                    continue
                if minBinPcnt > 0:
                    if df1_ks['total'].sum() * 1.0 / N < minBinPcnt or df2_ks['total'].sum() * 1.0 / N < minBinPcnt:
                        continue
                # 将切分点的值加入列表
                cutoffpoints.append(cutoff_value)
                # 每箱样本数量大于1，可以再分箱
                if df1.shape[0] > 1:
                    df_temp.append(df1)
                if df2.shape[0] > 1:
                    df_temp.append((df2))
            df_list = df_temp

        # 由于是二叉树型切分，所以切分点个数可能不满足要求，故需判断，再做处理
        while len(cutoffpoints) > max_points:
            regroup_ks = KS(df_clean, col, target)
            regroup_temp = regroup_ks[regroup_ks[col].isin(cutoffpoints)]
            min_ks = minAbsValue(regroup_temp['ks'].values.tolist())
            col_value = regroup_temp[regroup_temp['ks'] == min_ks][col].values[0]
            cutoffpoints.remove(col_value)
        # 检验分箱后单调性
        # df_clean[col+'lv'] = df_clean[col].apply(lambda x: assign_group(x, cutoffpoints))
        # is_monotonous = BadRateMontone2(df_clean, col+'lv', target)
        # while is_monotonous:
        #     max_points -= 1
        if special_attr is not None:
            cutoffpoints = special_attr + sorted(cutoffpoints)
        else:
            cutoffpoints = sorted(cutoffpoints)
    return cutoffpoints

def assign_bin2(x, cut_point, special_attr=None):
    if special_attr is not None:
        if x in special_attr:
            return str(x)
    num_Bins = len(cut_point)
    cut_point.sort()
    if x <= cut_point[0]:
        return '(,{}]'.format(cut_point[0])
    elif x > cut_point[-1]:
        return '({},)'.format(cut_point[-1])
    else:
        for i in range(0, num_Bins - 1):
            if cut_point[i] < x <= cut_point[i + 1]:
                return '({},{}]'.format(cut_point[i], cut_point[i + 1])

## 判断某变量的坏样本率是否单调
def BadRateMontone(df, col, target, special_attr=None):
    '''判断坏样本率是否单调'''
    if special_attr is not None:
        df_clc = df.loc[~df[col].isin(special_attr)]
    else:
        df_clc = df.copy()
    if len(set(df_clc[col])) <= 2:
        return True
    regroup = BadRate(df_clc, col, target).sort_values(by=col)
    bad_rate_diff = regroup['bad_rate'].diff().values
    temp = bad_rate_diff[1:] > 0
    if sum(temp) != len(temp) or sum(temp) != 0:
        return True
    else:
        return False

#皮尔森相关系数
def pearson(vector1, vector2):
    '''
    计算两个向量之间的皮尔森相关系数，皮尔森相关系数是衡量线性关联性的程度，p的一个几何解释是其代表两个变量的取值根据均值集中后构成的向量之间夹角的余弦。
    :param vector1: 向量1
    :param vector2: 向量2
    :return: 返回皮尔森相关系数值
    '''
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

# 基尼系数
def gini_coef(wealths):
    '''
    计算基尼系数
    :param wealths: 序列，一串数字
    :return: 返回基尼系数值
    '''
    #从0开始计算数组累计值
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    # 获得原数组的和，即为累计值数组的最后一个
    sum_wealths = cum_wealths[-1]
    # 人数累计占比
    xarray = np.array(np.arange(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    # 收入累计占比
    yarray = cum_wealths / sum_wealths
    # 用积分方法计算曲线下的面积
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A+B)

# 洛仑兹曲线
def lorenz_curve(wealths):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplot()
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(np.arange(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
    upper = xarray
    yarray = cum_wealths / sum_wealths
    ax.plot(xarray, yarray)
    ax.plot(xarray, upper)
    ax.set_xlabel(u'人数累积占比')
    ax.set_xlabel(u'收入累积占比')
    plt.show()

# 信息增益
def info_gain(df, col, target):
    '''
    计算信息增益
    :param df: 数据框
    :param col: 字段或变量
    :param target: 目标变量，0-好，1-坏
    :return: 返回信息增益值
    '''
    temp_df = df.copy()
    regroup_df = temp_df[[col, target]].groupby([col, target])[col].count().unstack()
    regroup_df['sum'] = regroup_df.apply(lambda x: x.sum(), axis=1)
    regroup_df.loc['total'] = regroup_df.apply(lambda x: x.sum())

    regroup_df['bad_rate'] = regroup_df.apply(lambda x: x[1] / x['sum'], axis=1)
    regroup_df['good_rate'] = regroup_df.apply(lambda x: x[0] / x['sum'], axis=1)

    regroup_df['entropy'] = regroup_df.apply(lambda x: -x['bad_rate']*math.log2(x['bad_rate'])-x['good_rate']*math.log2(x['good_rate']), axis=1)
    N = regroup_df.loc['total']['sum']
    regroup_df['cond_entropy'] = regroup_df.apply(lambda x: x['sum'] / N * x['entropy'], axis=1)
    return regroup_df['cond_entropy'][-1] - regroup_df['cond_entropy'][:-1].sum()

#卡方分布值
def chi_square(df, col, target):
    '''
    计算变量的卡方分布值
    :param df: 数据框
    :param col: 字段或变量
    :param target: 目标变量，0-好，1-坏
    :return: 返回卡方分布值
    '''
    temp_df = df.copy()
    regroup_df = temp_df[[col, target]].groupby([col, target])[col].count().unstack()
    regroup_df.fillna(0, inplace=True)
    regroup_df['sum'] = regroup_df.apply(lambda x: x.sum(), axis=1)
    regroup_df.loc['total'] = regroup_df.apply(lambda x: x.sum())
    N = regroup_df.loc['total']['sum']
    good_rate = regroup_df.loc['total'][0] * 1.0 / N
    bad_rate = regroup_df.loc['total'][1] * 1.0 / N
    regroup_df['expected_good'] = regroup_df.apply(lambda x: x['sum'] * good_rate, axis=1)
    regroup_df['expected_bad'] = regroup_df.apply(lambda x: x['sum'] * bad_rate, axis=1)
    regroup_df.drop(['total'], inplace=True)
    chi_good = [(av - ev) ** 2 / ev for ev, av in zip(regroup_df['expected_good'], regroup_df[0])]
    chi_bad = [(av - ev) ** 2 / ev for ev, av in zip(regroup_df['expected_bad'], regroup_df[1])]
    chisq = sum(chi_good) + sum(chi_bad)
    return chisq

# 克雷姆值
def cramerV(df, col, target):
    chisq = chi_square(df, col, target)
    N = df.shape[0]
    return np.sqrt(chisq/N)

# woe值和iv值
def woe_iv(df, col, target):
    '''
    计算woe值和iv值
    :param df: 数据框
    :param col: 要计算的字段或变量
    :param target: 目标变量，0-好，1-坏
    :return: 返回woe值和iv值
    '''
    # print("{} is in processing".format(col))
    temp_df = df.copy()
    total = temp_df[col].value_counts().reset_index().rename(columns={col: 'total', 'index': col})
    bad = temp_df.groupby(col)[target].sum().reset_index().rename(columns={target: 'bad'})
    regroup = pd.merge(total, bad, on=col)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup.total - regroup.bad
    regroup['pcnt_good'] = regroup['good'].apply(lambda x: x / (N - B))
    regroup['pcnt_bad'] = regroup['bad'].apply(lambda x: x / B)
    eps = 1e-10
    regroup['WOE'] = regroup.apply(lambda x: np.log((x.pcnt_good+eps) / (x.pcnt_bad+eps)), axis=1)
    regroup['IV'] = regroup.apply(lambda x: (x.pcnt_good - x.pcnt_bad) * x.WOE, axis=1)

    dict_woe = {k: v for k, v in zip(regroup[col], regroup['WOE'])}
    IV = regroup['IV'].sum()
    return {"woe": dict_woe, "iv": IV}

def get_overdueDays(end_date, practical_date, given_date):
    '''
    计算逾期天数
    :param end_date: 约定还款日期
    :param practical_date: 实际还款日期
    :param given_date: 给定日期，当practical_date为空值或-1时，采用给定的日期来计算逾期天数
    :return: 天数
    '''
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    if pd.isnull(practical_date) or practical_date == -1:
        if isinstance(given_date, str):
            given_date = datetime.datetime.strptime(given_date, '%Y-%m-%d %H:%M:%S')
        return (given_date - end_date).days
    else:
        if isinstance(practical_date, str):
            practical_date = datetime.datetime.strptime(practical_date, '%Y-%m-%d %H:%M:%S')
        return (practical_date - end_date).days


def batch_analysis(df, col, target):
    print('********************%s************************' % col)
    print(BadRate(df, col, target))
    print('*****************************************************************************************')
    print('IV值:', woe_iv(df, col, target)['iv'])
    print('克雷姆值:', cramerV(df, col, target))
    print('\n')

def MyAssignBin(x, cutOffPoints):
    '''
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :return: 分箱后的对应的第几个箱

    '''
    cutOffPoints.sort()
    numBin = len(cutOffPoints)
    if -1 in cutOffPoints:
        if x == -1:
            return '{}'.format(-1)
        elif x <= cutOffPoints[1]:
            return '<={}'.format(cutOffPoints[1])
        elif x > cutOffPoints[-1]:
            return '>{}'.format(cutOffPoints[-1])
        else:
            for i in range(1,numBin-1):
                if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                    return '({},{}]'.format(cutOffPoints[i], cutOffPoints[i+1])
    else:
        if x<=cutOffPoints[0]:
            return '<={}'.format(cutOffPoints[0])
        elif x > cutOffPoints[-1]:
            return '>{}'.format(cutOffPoints[-1])
        else:
            for i in range(0,numBin-1):
                if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                    return '({},{}]'.format(cutOffPoints[i], cutOffPoints[i+1])

def UniversalAssignBin(x, cutOffPoints):
    '''
    没有特殊值
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :return: 分箱后的对应的第几个箱

    '''
    cutOffPoints.sort()
    numBin = len(cutOffPoints)
    if x<=cutOffPoints[0]:
        return '<={}'.format(cutOffPoints[0])
    elif x > cutOffPoints[-1]:
        return '>{}'.format(cutOffPoints[-1])
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return '({},{}]'.format(cutOffPoints[i], cutOffPoints[i+1])

def BadRateEncoding(df, col, target):
    '''
    类别多值非数值变量编码
    '''
    regroup = BadRate(df, col, target)
    br_dict = regroup[[col, 'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = round(v['bad_rate'], 3)
    badRateEncoding = df[col].map(lambda x: br_dict[x])
    return {'encoding': badRateEncoding, 'bad_rate': br_dict}

def function_binning(df, col, target, method='frequency'):
    '''
    分箱，计算切分点、iv值、克雷姆值
    :param df: 数据框
    :param col: 需要处理的变量
    :param target: 目标变量, 0-好,1-坏
    :param method: 分箱方法，frequency-等频分箱,chisq-卡方分箱
    :return: 变量名称，切分点，iv值，克雷姆值
    '''
    print('{} is in processing'.format(col))
    values_col = list(set(df[col]))
    cutoffPoints = []
    if -1 in values_col:
        if method == 'frequency':
            cutoffPoints = equal_frequency_binning(df, col, target, special_attr=[-1])[1]
            cutoffPoints += [-1]
        if method == 'chisq':
            cutoffPoints = cutoff_bin(df, col, target)[col]
        df[col + '_Bin'] = df[col].apply(lambda x: assign_bin(x, cutoffPoints, special_attr=[-1]))
        iv = woe_iv(df, col + '_Bin', target)['iv']
        cramer = cramerV(df, col + '_Bin', target)
        return {'col': col, 'cutoffPoints': cutoffPoints, 'iv': iv, 'cramer': cramer}
    else:
        if method == 'frequency':
            cutoffPoints = equal_frequency_binning(df, col, target)[1]
        if method == 'chisq':
            cutoffPoints = cutoff_bin(df, col, target)[col]
        df[col + '_Bin'] = df[col].apply(lambda x: assign_bin(x, cutoffPoints))
        iv = woe_iv(df, col + '_Bin', target)['iv']
        cramer = cramerV(df, col + '_Bin', target)
        return {'col': col, 'cutoffPoints': cutoffPoints, 'iv': iv, 'cramer': cramer}

def correlation_test(df, IV_dict_sorted):
    '''
    两两间的线性相关性检验
    1，将候选变量按照IV进行降序排列
    2，计算第i和第i+1的变量的线性相关系数
    3，对于系数超过阈值的两个变量，剔除IV较低的一个
    '''
    num_variable = len(IV_dict_sorted)
    deleted_variable = []
    for i in range(num_variable-1):
        name1 = IV_dict_sorted[i][0] + '_WOE'
        if name1 in deleted_variable:
            continue
        for j in range(i+1, num_variable):
            name2 = IV_dict_sorted[j][0] + '_WOE'
            if name2 in deleted_variable:
                continue
            roh = np.corrcoef(df[name1], df[name2])[0, 1]
            if abs(roh)>0.7:
                iv1 = IV_dict_sorted[i][1]
                iv2 = IV_dict_sorted[j][1]
                if iv1 > iv2:
                    deleted_variable.append(name2)
                else:
                    deleted_variable.append(name1)
    multi_analysis_vars = [i[0]+'_WOE' for i in IV_dict_sorted if i[0]+'_WOE'not in deleted_variable]
    return multi_analysis_vars


def logit_reg(df, variables_list, target, random_state=0):
    userData = df[variables_list + [target]].copy()
    # 划分训练集和测试集
    x1, x0 = userData[userData[target] == 1], userData[userData[target] == 0]
    y1, y0 = x1[target], x0[target]
    x1.drop([target], axis=1, inplace=True)
    x0.drop([target], axis=1, inplace=True)
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=random_state, test_size=0.3)
    x0_train, x0_test, y0_train, y0_test = train_test_split(x0, y0, random_state=random_state, test_size=0.3)

    x_train, x_test = pd.concat([x1_train, x0_train]), pd.concat([x1_test, x0_test])
    y_train, y_test = pd.concat([y1_train, y0_train]), pd.concat([y1_test, y0_test])
    # 添加常数项
    x_train = sm.add_constant(x_train)
    x_test = sm.add_constant(x_test)

    LR = sm.Logit(y_train, x_train).fit(disp=False)
    # summary_ = LR.summary()
    pvals = LR.pvalues.to_dict()
    # 逐步剔除不显著的变量
    varLargeP = {k: v for k, v in pvals.items() if v > 0.05}
    varLargeP = sorted(varLargeP.items(), key=lambda x: x[1], reverse=True)
    while (len(varLargeP) > 0 and len(variables_list) > 0):
        varMaxP = varLargeP[0][0]
        # print(varMaxP)
        variables_list.remove(varMaxP)

        x_train = x_train[variables_list]
        x_train = sm.add_constant(x_train)

        x_test = x_test[variables_list]
        x_test = sm.add_constant(x_test)

        LR = sm.Logit(y_train, x_train).fit(disp=False)
        # summary_ = LR.summary()
        pvals = LR.pvalues.to_dict()
        varLargeP = {k: v for k, v in pvals.items() if v > 0.05}
        varLargeP = sorted(varLargeP.items(), key=lambda x: x[1], reverse=True)
    for k, v in LR.params.to_dict().items():
        if v > 0:
            variables_list.remove(k)
    x_train = x_train[variables_list]
    x_train = sm.add_constant(x_train)

    x_test = x_test[variables_list]
    x_test = sm.add_constant(x_test)

    LR = sm.Logit(y_train, x_train).fit(disp=False)
    return LR, x_train, y_train, x_test, y_test, variables_list

def write_xlsx(filePath, df, index=False, sheet_name='Sheet1'):
    '''写入excel文件'''
    with pd.ExcelWriter(filePath) as writer:
        df.to_excel(writer, index=index, sheet_name=sheet_name)

def read_csv(file_path):
    '''读取csv文件'''
    try:
        data = pd.read_csv(file_path)
    except OSError:
        '''读取带中文路径的csv文件'''
        with open(file_path,encoding='utf-8') as f:
            data = pd.read_csv(f)
    return data


def cal_psi(expected_result, actual_result, N=10):
    expected_result.sort()  # 排序
    minValue, maxValue = min(expected_result), max(expected_result)
    interval = (maxValue - minValue) / N
    splitPoints = [interval * i for i in range(1, N)]
    splitPoints.sort()
    expected_dcit = {}
    expected_dcit['<={}'.format(splitPoints[0])] = round(
        sum([v <= splitPoints[0] for v in expected_result]) / len(expected_result), 4)
    expected_dcit['>{}'.format(splitPoints[-1])] = round(
        sum([v > splitPoints[-1] for v in expected_result]) / len(expected_result), 4)
    for i in range(N - 2):
        expected_dcit['({},{}]'.format(splitPoints[i], splitPoints[i + 1])] = round(
            sum([v <= splitPoints[i + 1] and v > splitPoints[i] for v in expected_result]) / len(expected_result), 4)

    actual_dcit = {}
    actual_dcit['<={}'.format(splitPoints[0])] = round(
        sum([v <= splitPoints[0] for v in actual_result]) / len(actual_result), 4)
    actual_dcit['>{}'.format(splitPoints[-1])] = round(
        sum([v > splitPoints[-1] for v in actual_result]) / len(actual_result), 4)
    for i in range(N - 2):
        actual_dcit['({},{}]'.format(splitPoints[i], splitPoints[i + 1])] = round(
            sum([v <= splitPoints[i + 1] and v > splitPoints[i] for v in actual_result]) / len(actual_result), 4)

    expected_df = pd.DataFrame(expected_dcit, index=['expected_rate']).T
    expected_df.index.name = 'prob_section'
    actual_df = pd.DataFrame(actual_dcit, index=['actual_rate']).T
    actual_df.index.name = 'prob_section'
    expected_actual_df = pd.merge(expected_df, actual_df, left_index=True, right_index=True)
    expected_actual_df['psi'] = expected_actual_df.apply(
        lambda x: (x.actual_rate - x.expected_rate) * (np.log(x.actual_rate / x.expected_rate)), axis=1)
    return expected_actual_df['psi'].sum()

def get_roc_curve(y_true, y_pred_prob):
    fpr,tpr,thresholds = roc_curve(y_true,y_pred_prob)
    roc_auc = roc_auc_score(y_true,y_pred_prob)
    #画ROC曲线
    fig, ax = plt.subplots(figsize=(6, 6))
    font1 = {'family' : 'Times New Roman',
             'weight' : 'normal',
             'size'   : 15,
            }
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr,tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.legend(loc="lower right", prop=font1)
    plt.xlabel('FPR', font1)
    plt.ylabel('TPR', font1)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.show()

#ks函数
get_ks = lambda df, y_pred,y_true: ks_2samp(df[df[y_true]==1][y_pred], df[df[y_true]!=1][y_pred]).statistic

def read_json(jsonFile):
    '''
    读取json文件
    :param jsonFile: 要读取的json文件
    :return: json数据
    '''
    with open(jsonFile, 'r', encoding='utf-8') as f:
        return json.load(f)

def show_iv_feature(IV_dict_sorted):
    #画出iv值分布
    IV_values = [iv[1] for iv in IV_dict_sorted]
    IV_names = [iv[0] for iv in IV_dict_sorted]
    plt.title('feature IV')
    plt.bar(range(len(IV_names)), IV_values)
    plt.yticks(np.linspace(0, max(IV_values)+0.02, 20))
    plt.grid(True)
    plt.show()

def show_variable_corr(df, IV_names_new):
    #计算相关系数矩阵，并画出热力图进行数据可视化
    df_WOE = df[IV_names_new]
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df_WOE.corr()
    cmap = sns.diverging_palette(240, 10, n=len(IV_names_new), as_cmap=True)
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=cmap, square=True, ax=ax)
    plt.show()
