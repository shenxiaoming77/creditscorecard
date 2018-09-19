#coding=utf-8

import  pandas as pd
import  numpy as np

from  settings import  *
from  util.featureEngineering_functions import *


'''
cross feature:
register_apply_date_interval
'''
def register_apply_date_interval(derivation_df):
    length = len(train_df['apply_date'])
    apply_date = list(train_df['apply_date'])
    register_date = list(train_df['register_date'])

    date_interval = []
    for i in range(length):
        date1 = register_date[i]
        date2 = apply_date[i]
        date_interval.append(days(str(date2), str(date1)))

    derivation_df['register_apply_date_interval'] = date_interval



def jobLevel_cross_applyDatehour(job, apply_hour):
    job_list = ["专业技术人员/设计师/工程师", "主任/主管/组长/初级管理", "学生", "总监/总经理/高管",
                "普通员工","经理/中级管理","销售人员"]
    hour = int(apply_hour)
    if job in job_list and hour >= 0 and hour <= 6:
        return  1
    else:
        return  0


'''
cross feature:
以下的特定的job_level岗位在凌晨0点到6之前申请贷款，坏账率呈现相对较高的趋势，这类人在凌晨的apply
看成是疑似异常贷款申请：seemingly_abnormity_application

专业技术人员/设计师/工程师    0.272727
主任/主管/组长/初级管理     0.197605
学生                0.120000
总监/总经理/高管         0.111111
普通员工              0.220800
经理/中级管理           0.151261
销售人员              0.277778
'''
def seemingly_abnormity_application(derivation_df):
    derivation_df['seemingly_abnormity_application'] = train_df.apply(lambda row : jobLevel_cross_applyDatehour(row
["job_level"], row["applyDate_hour"]), axis = 1)

#城市名称衍生出城市等级变量
def identity_city_classification(derivation_df):
    with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as file:
         city_level_dict = pickle.load(file)
    derivation_df['identity_city_classification'] = train_df['identity_city']\
        .apply(lambda x : assign_city_level_bin(x, city_level_dict))
    file.close()

#城市名称衍生出城市等级变量
def phone_city_classification(derivation_df):
    with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as file:
         city_level_dict = pickle.load(file)
    derivation_df['phone_city_classification'] = train_df['phone_city']\
        .apply(lambda x : assign_city_level_bin(x, city_level_dict))
    file.close()

if __name__ == '__main__':
    train_df = pd.read_excel(ROOT_DIR + 'train.xlsx', encoding='utf-8')
    derivation_df = pd.DataFrame()
    derivation_df['user_id'] = list(train_df['user_id'])

    # register_apply_date_interval(derivation_df)
    # seemingly_abnormity_application(derivation_df)

    identity_city_classification(derivation_df)
    phone_city_classification(derivation_df)

    print(derivation_df)


