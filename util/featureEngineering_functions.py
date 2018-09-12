#coding=utf-8

import  numpy as np
import  pandas as pd
import  pickle
from  settings import  *

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
    reg_city_name = city_name.replace("市", "")
    if reg_city_name.find('null') >= 0:
        return  'null'

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



if __name__ == '__main__':
    #assign_city_level_bin('杭州市')
    with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as file:
        city_level_dict = pickle.load(file)
    print(city_level_dict)
    print(assign_city_level_bin('null', city_level_dict))