#coding=utf-8

import  pandas as pd
import  numpy as np
from  util.featureEngineering_functions import *

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


if __name__ == '__main__':
    user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')
    job_level_combine(user_info_df)
