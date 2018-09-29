#coding=utf-8

import  pandas as pd
import  numpy as np
from  util.scorecard_functions import *

#job_level中的几个类别特征值合并到对应的大类当中
#降低特征值的冗余
'''
'主管' ---> '主任/主管/组长/初级管理'
'总监', '总经理' ---> '总监/总经理/高管'
'经理'---> '经理/中级管理',
'''
def job_level_combine(df):
    df['job_level'] = df['job_level'].apply(lambda x : job_level_combine_func(x))
    print(set(df['job_level']))
    print(len(set(df['job_level'])))

#network_len中的几个时间段存在重合，进行合并
def network_len_combine(df):
    df['network_len'] = df['network_len'].apply(lambda  x : network_len_combine_func(x))

if __name__ == '__main__':
    user_info_df = pd.read_excel(ROOT_DIR + 'preProcessed_user_info.xlsx', encoding='utf-8')
    job_level_combine(user_info_df)
    #network_len_combine(user_info_df)
    #print(set(user_info_df['network_len']))


