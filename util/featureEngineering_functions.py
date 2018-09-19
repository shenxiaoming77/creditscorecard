#coding=utf-8

import  numpy as np
import  pandas as pd
import  pickle
from  settings import  *
import  datetime
from  math import  isnan

def loadFeatures(file):

    lines = []
    try:
        with open(file) as f:
            lines_object = f.readlines()

        for object in lines_object:
            lines.append(object.strip())

    finally:
        return  lines

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

#划定城市的等级，确定属于哪个层级
def assign_city_level_bin(city_name, city_level_dict):
    print(str(city_name))
    if str(city_name).find('nan') >= 0:
        return  '其他'

    reg_city_name = city_name.replace("市", "")

    keys = city_level_dict.keys()
    flag = False
    for key in keys:
        values = list(city_level_dict[key])

        for value in values:
            if str(value).find(reg_city_name) >= 0:
                return  key

    if flag == False:
        print(reg_city_name)
        return  '其他'

#对年龄进行区间划分
def assign_age_bin(x):
    if 18<=x<=20:
        return 0
    elif 20<x<=35:
        return 1
    elif 35<x<=50:
        return 2
    elif 50<x<=60:
        return 3

#芝麻信用分等级划分
#第一级：300分~500分
#第二级：500分~600分
#第三级：600分~700分
#第四级：700分~800分
#第五级：800分~950分
def assign_zhimaScore_bin(x):
    score = int(x)
    if  300 < score <= 500:
        return  0
    elif 500 < score <= 600:
        return 1
    elif 600 < score <= 700:
        return  2
    elif 700 < score <= 800:
        return  3
    elif score > 900:
        return  4
    else:
        return  -1

#百融信用分
#[300,500) 高风险，建议拒绝
#[500,550) 中风险，建议关注
#[550,1000] 低风险风险，建议通过
def assign_brScore_bin(x):
    score = int(x)
    if 300 <= score < 500:
        return 0
    elif 500 <= score < 550:
        return  1
    elif score >= 550:
        return  2
    else:
        return  -1

def days(str1,str2):
    date1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d")
    date2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d")
    num =(date1-date2).days
    return num

def job_level_combine_func(job_level):
    if job_level == "主管":
        return  "主任/主管/组长/初级管理"
    elif job_level == "总监" or job_level == "总经理":
        return "总监/总经理/高管"
    elif job_level == "经理":
        return  "经理/中级管理"
    else:
        return  job_level

def network_len_combine_func(networkLen):
    length = str(networkLen)
    if length == '(3,6]':
        return  '(0,6]'
    elif length == '6':
        return  '(0,6]'
    elif length == '未查得':
        return  'missing'
    else:
        return  length


if __name__ == '__main__':
    #assign_city_level_bin('杭州市')
    # with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as file:
    #     city_level_dict = pickle.load(file)
    # print(city_level_dict)
    # print(assign_city_level_bin('null', city_level_dict))

    train_df = pd.read_excel(ROOT_DIR + 'train.xlsx', encoding = 'utf-8')
    length = len(train_df['apply_date'])


    for i in range(length):
        date1 = list(train_df['register_date'])[i]
        date2 = list(train_df['apply_date'])[i]

        print(days(str(date2), str(date1)))
