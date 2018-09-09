#coding=utf-8

import  numpy as np

from  settings import  city_level_code_dict

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
def assign_city_level_bin(city_name):
    city_name = city_name.replace('市', '')

    if city_name.find('null') >= 0:
        return  'null'




if __name__ == '__main__':
    assign_city_level_bin('杭州市')