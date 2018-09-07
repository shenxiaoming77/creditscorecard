#coding = utf-8

import  pandas as pd
from sklearn.model_selection import  train_test_split
from  settings import *


data = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding = 'utf-8')

features_list = numericalFeatures + categoricalFeatures

label = 'loan_status'

X = data[features_list]
y = data[label]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, shuffle=True, stratify=y)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
'''
train_sum = train_y.sum()
test_sum = test_y.sum()

print(train_sum)
print(test_sum)

print(train_sum / 12474.0)
print(test_sum / 5347.0)
'''

numerical_data = data[numericalFeatures]
categorical_data = data[categoricalFeatures]

numerical_data.fillna('null', inplace = True)
categorical_data.fillna('null', inplace = True)

train = pd.concat([train_x, train_y], axis = 1)
test = pd.concat([test_x, test_y], axis = 1)
train.fillna('null')
test.fillna('null')



numerical_data.to_excel(ROOT_DIR + 'numerical_data.xlsx', index=None, encoding='utf-8')
categorical_data.to_excel(ROOT_DIR + 'categorical_data.xlsx' , index = None, encoding='utf-8')


train.to_excel(ROOT_DIR + 'train.xlsx',index= None, encoding='utf-8')
test.to_excel(ROOT_DIR + 'test.xlsx', index= None, encoding='utf-8')