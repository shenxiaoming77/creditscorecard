#coding=utf-8

import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  pickle
import  seaborn as sns

from  settings import  *
from util.scorecard_functions import *

user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')


#所有原始特征的空值情况统计
def null_counts_analysis():
    null_count_dict = {}
    total = len(user_info_df['job_level'])
    features_list = [x for x in user_info_df.columns if x not in ['user_id', 'loan_status', 'bill_id',
                                                                  'phone','name','identity']]

    for var in features_list:
        null_count = len(user_info_df[var]) - user_info_df[var].count()
        null_count_dict[var] = null_count
        if null_count > 0:
            print(var)

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
    plt.xticks(np.arange(len(x)), x)
    plt.bar(np.arange(len(x)), y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/null_count.png')
    plt.show()

#分析芝麻分与其他连续变量的相关性，根据强相关特征来进行用户分群，从而对同组用户进行芝麻分的缺失填补
def zhima_score_corrcoef_analysis():

    user_info_df.fillna(-1, inplace = True)
    numerical_data = user_info_df[FEATURE_DICT['numericalFeatures']]

    x = 'zhima_score'

    x_data = numerical_data[x].astype('int')
    '''
    zhima_score与auth_level, company_college_length 这两个连续变量关联度较高
    '''
    corrcoef_dict = {}
    for y in FEATURE_DICT['numericalFeatures']:
        print(y)
        if x != y:
            y_data = numerical_data[y].astype('int')
            print(x, '      ', y, '     ', np.corrcoef(x_data, y_data)[0, 1])
            corrcoef_value = np.corrcoef(x_data, y_data)[0, 1]
            if corrcoef_value >= 0.1:
                corrcoef_dict[y] = corrcoef_value

    sorted_corrcoef_dict = sorted(corrcoef_dict.items(), key = lambda d : d[1], reverse = True)
    print(sorted_corrcoef_dict)

    features = []
    for item in sorted_corrcoef_dict:
        features.append(item[0])
        print(item[0])

    features_df = pd.DataFrame()
    features_df['feature'] = features
    features_df.to_excel(ROOT_DIR + 'featureEngineering/zhima_correlation_features.xlsx', index=None)

#观察芝麻分缺失的用户的群体特征情况
def zhima_score_missing_analysis():

    data_df = user_info_df[user_info_df['zhima_score'].isnull()]
    job_level_count = data_df.groupby('job_level')['job_level'].count()

    print(job_level_count)




#原始连续特征之间的相关性热力图分析
def origin_numericalFeature_correlation_analysis():
    numerical_df = user_info_df[FEATURE_DICT['numericalFeatures']]
    numerical_df.fillna(-1, inplace = True)

    corr_mat = numerical_df.corr()

    sns.set()
    f, ax = plt.subplots(figsize=(9, 10))

    #使用不同的颜色
    sns.heatmap(corr_mat, fmt="d",cmap='YlGnBu', ax=ax)

    ax.set_title('correlations analysis for numerical features')


    #设置坐标字体方向
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')



    plt.savefig('./visualization_pics/correlations_analysis_numericalFeatures.png')
    plt.show()


def phone_province_analysis():
    user_info_df['phone_province'].fillna('missing', inplace = True)

    phone_province_groupby = user_info_df.groupby('phone_province')[LABEL]
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
    sorted_dict = sorted(badrate_dict.items(), key = lambda  d : d[1], reverse = False)

    with open(ROOT_DIR + 'dataExploration/phone_province_badrate.pkl', 'wb') as file:
        pickle.dump(sorted_dict, file)

    x = []
    y = []
    for item in sorted_dict:
        x.append(item[0])
        y.append(item[1])
    x_label='phone province'
    y_label= 'badrate'
    title='手机号所在省份的整体坏账率'
    plt.bar(np.arange(len(x)), y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(np.arange(len(x)), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/phone_province_badrate.png')
    plt.show()

def identity_province_analysis():
    user_info_df['identity_province'].fillna('missing', inplace = True)

    identity_province_groupby = user_info_df.groupby('identity_province')[LABEL]
    groupby_count = identity_province_groupby.count()
    groupby_sum = identity_province_groupby.sum()


    count_list = list(groupby_count)
    sum_list = list(groupby_sum)
    keys = list(dict(list(identity_province_groupby)).keys())
    print(keys)
    badrate_dict = {}
    for i in range(len(keys)):
        badrate_dict[keys[i]] = (sum_list[i] * 1.0 / count_list[i])
        print(keys[i], '    ', sum_list[i], '   ', count_list[i])

    #升序排列
    sorted_dict = sorted(badrate_dict.items(), key = lambda  d : d[1], reverse = False)

    with open(ROOT_DIR + 'dataExploration/identity_province_badrate.pkl', 'wb') as file:
        pickle.dump(sorted_dict, file)

    x = []
    y = []
    for item in sorted_dict:
        x.append(item[0])
        y.append(item[1])
    x_label='identity province'
    y_label= 'badrate'
    title='户籍所在省份的整体坏账率'
    plt.bar(np.arange(len(x)), y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(np.arange(len(x)), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/identity_province_badrate.png')
    plt.show()

def phone_city_analysis():
    user_info_df['phone_city'].fillna('null', inplace = True)
    city_classification_dict = {}
    with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as  file:
        city_classification_dict = pickle.load(file)
    user_info_df['phone_city_bin'] = user_info_df['phone_city'].apply(lambda  x : assign_city_level_classification(x, city_classification_dict))

    phone_city_groupby = user_info_df.groupby('phone_city_bin')[LABEL]
    bin_count = phone_city_groupby.count()
    bin_badstatus_sum = phone_city_groupby.sum()

    count_list = list(bin_count)
    sum_list = list(bin_badstatus_sum)
    keys = list(dict(list(phone_city_groupby)).keys())

    badrate_dict = {}
    for i in range(len(keys)):
        badrate_dict[keys[i]] = (sum_list[i] * 1.0) / count_list[i]

    sorted_dict = sorted(badrate_dict.items(), key = lambda  d : d[1], reverse = False)

    with open(ROOT_DIR + 'dataExploration/phone_city_badrate.pkl', 'wb') as file:
        pickle.dump(sorted_dict, file)

    x = []
    y = []
    for item in sorted_dict:
        x.append(item[0])
        y.append(item[1])

    x_label='phone city level'
    y_label= 'badrate'
    title='号码归属地城市等级的整体坏账率'
    plt.bar(np.arange(len(x)), y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(np.arange(len(x)), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/phone_city_badrate.png')
    plt.show()

def identity_city_classification_analysis():
    user_info_df['identity_city'].fillna('null', inplace = True)
    city_classification_dict = {}
    with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as  file:
        city_classification_dict = pickle.load(file)
    user_info_df['identity_city_bin'] = user_info_df['identity_city']\
        .apply(lambda  x : assign_city_level_classification(str(x), city_classification_dict))
    print(user_info_df['identity_city_bin'])


    identity_city_groupby = user_info_df.groupby('identity_city_bin')[LABEL]
    bin_count = identity_city_groupby.count()   #每个bin的样本数量
    bin_badstatus_sum = identity_city_groupby.sum()   #每个bin的坏样本数量

    count_list = list(bin_count)
    sum_list = list(bin_badstatus_sum)
    keys = list(dict(list(identity_city_groupby)).keys())

    badrate_dict = {}
    for i in range(len(keys)):
        badrate_dict[keys[i]] = (sum_list[i] * 1.0) / count_list[i]

    #升序排列
    sorted_dict = sorted(badrate_dict.items(), key = lambda  d : d[1], reverse = False)

    with open(ROOT_DIR + 'dataExploration/identity_city_classification_badrate.pkl', 'wb') as file:
        pickle.dump(sorted_dict, file)

    x = []
    y = []
    for item in sorted_dict:
        x.append(item[0])
        y.append(item[1])

    x_label='identity city classification'
    y_label= 'badrate'
    title='户籍所在城市等级的整体坏账率'
    plt.bar(np.arange(len(x)), y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(np.arange(len(x)), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/identity_city_classification_badrate.png')
    plt.show()


def message_feature_analysis():
    messageFeature_list = ['m1_verif_count','m1_register_count','m1_apply_request_count','m1_apply_reject_count',
                'm1_loan_offer_count','m1_repay_fail_count','m1_overdue_count','m1_repay_remind_count',
                'm3_verif_count','m3_register_count','m3_apply_request_count','m3_apply_reject_count',
                'm3_loan_offer_count','m3_repay_fail_count','m3_overdue_count','m3_repay_remind_count',
                'm6_verif_count','m6_register_count','m6_apply_request_count','m6_apply_reject_count',
                'm6_loan_offer_count','m6_repay_fail_count','m6_overdue_count','m6_repay_remind_count',
                'm12_verif_count','m12_register_count','m12_apply_request_count','m12_apply_reject_count',
                'm12_loan_offer_count','m12_repay_fail_count','m12_overdue_count','m12_repay_remind_count'
                ]

def network_len_analysis():
    network_len_df = user_info_df[['network_len', 'loan_status']]
    network_len_df.fillna('missing', inplace=True)
    network_len_df['network_len_tmp'] = network_len_df['network_len'].apply(lambda x : str(x).replace("未查得", "null"))
    network_len_df.drop('network_len', inplace = True, axis = 1)
    network_len_df['network_len'] = network_len_df['network_len_tmp']
    network_len_df.drop('network_len_tmp', inplace = True, axis = 1)

    print(network_len_df)

    groupby_data = network_len_df.groupby('network_len')[LABEL]
    bin_count = groupby_data.count()
    bin_badstatus_sum = groupby_data.sum()

    count_list = list(bin_count)
    sum_list = list(bin_badstatus_sum)
    keys = list(dict(list(groupby_data)).keys())

    badrate_dict = {}
    for i in range(len(keys)):
        badrate_dict[keys[i]] = (sum_list[i] * 1.0) / count_list[i]

    sorted_dict = sorted(badrate_dict.items(), key = lambda  x : x[1], reverse = False)

    with open(ROOT_DIR + 'dataExploration/network_len_badrate.pkl', 'wb') as file:
        pickle.dump(sorted_dict, file)

    x = []
    y = []
    for item in sorted_dict:

        x.append(item[0])
        y.append(item[1])

    x_label = 'network len year'
    y_label = 'badrate'

    print(x)
    print(y)

    title='入网时间不同区间的整体坏账率'
    plt.bar(np.arange(len(x)), y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(np.arange(len(x)), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/network_len_badrate.png')
    plt.show()

#异常值
def outlier_analysis():

    zhima_score_df = pd.DataFrame(user_info_df[['zhima_score', 'user_age','auth_level']])
    zhima_score_df.columns = ['zhima_score', 'user_age','auth_level']

    zhima_score_df.fillna(-1, inplace = True)
    print(zhima_score_df)
    zhima_score_df.boxplot()  # 这里，pandas自己有处理的过程，很方便哦。
    plt.ylabel("ylabel")
    plt.xlabel("xlabel")  # 我们设置横纵坐标的标题。
    plt.show()
    #plt.show()


#关联交叉分析: 注册时间与申请时间之间的时间间隔的badrate分布情况
def register_apply_date_interval_analysis():
    length = len(user_info_df['apply_date'])
    apply_date = user_info_df['apply_date']
    register_date = user_info_df['register_date']

    df = pd.DataFrame()
    df['user_id'] = user_info_df['user_id']
    df['loan_status'] = user_info_df['loan_status']

    date_interval = []
    for i in range(length):
        date1 = register_date[i]
        date2 = apply_date[i]
        date_interval.append(days(str(date2), str(date1)))

    df['interval'] = date_interval

    groupby_data = df.groupby('interval')
    bin_count = groupby_data.count()
    bin_badstatus_sum = groupby_data.sum()

#查看类别变量的bin的样本占比情况,对于样本量占比特别小的bin 需要合并到其他bin中


if __name__ == '__main__':
    #null_counts_analysis()

    #phone_province_analysis()

    identity_province_analysis()

    #phone_city_analysis()

    #identity_city_classification_analysis()

    #message_feature_analysis()

    #network_len_analysis()

    #origin_numericalFeature_correlation_analysis()

    #zhima_score_corrcoef_analysis()

    #zhima_score_missing_analysis()

    #outlier_analysis()