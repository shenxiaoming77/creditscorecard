#coding=utf-8

import  pandas as pd

from  util.basic.scorecard_functions import *
from  util.basic.scorecard_methods import *


rootDir = 'D:/conf_test/jiguang_test/'

user_data = pd.read_excel(rootDir + 'user_data_20181023.xlsx')
test_data = pd.read_excel(rootDir + 'test_data_20181026.xlsx')

merge_data = pd.merge(user_data, test_data, how = 'left', left_on = 'REQUEST_IDENTITY', right_on = 'IDENTITY')

phone_featurev2 = pd.read_excel(rootDir + 'v2/东方星空imei_featurev2测试结果_20181102.xlsx')
#print (phone_featurev2.head(10))

merge_data = pd.merge(merge_data, phone_featurev2, how='left', left_on= 'REQUEST_IDENTITY', right_on= 'IDENTITY')

var = 'app0273'
merge_data = merge_data[merge_data['is_overdue']>=0]
merge_data.fillna(-1, inplace=True)
#print(merge_data.columns)
target = 'is_overdue'
print('app0273 values:')
print(set(merge_data['app0273']))
dict_temp = function_binning(merge_data, var, target, method='chisq')