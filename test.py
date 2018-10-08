# dicts = {'a':'ff','b':'bb','1':'11','a':'aa',}
# print(list(dicts.keys()))
# print(list(dicts.values()))

from  settings import  *
import  pickle
import  pandas as pd
import  numpy as np
import  seaborn as sns
import  matplotlib.pyplot as plt
from  util.scorecard_functions import *

# df1 = pd.read_excel(ROOT_DIR + 'settings/1.xls', encoding='gbk')
#
#
# df2 = pd.read_excel(ROOT_DIR + 'settings/2.xls', encoding='gbk')
#
#
# df3 = pd.read_excel(ROOT_DIR + 'settings/3.xls', encoding='gbk')
#
#
# df4 = pd.read_excel(ROOT_DIR + 'settings/4.xls', encoding='gbk')
#
#
# df5 = pd.read_excel(ROOT_DIR + 'settings/5.xls', encoding='gbk')
#
# df6 = pd.read_excel(ROOT_DIR + 'settings/6.xls', encoding='gbk')
#
# city_dict = {}
# city_dict['一线'] = list(df1['一线'])
# city_dict['新一线'] = list(df2['新一线'])
# city_dict['二线'] = list(df3['二线']) + ['省直']
# city_dict['三线'] = list(df4['三线']) + ['淮安']
# city_dict['四线'] = list(df5['四线'])
# city_dict['五线'] = list(df6['五线']) + ['达州']
#
# print(city_dict)
#
# with open(ROOT_DIR + 'settings/city_classification.pkl', 'wb') as file:
#     pickle.dump(city_dict, file)
#
# with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as file:
#     city_dict = pickle.load(file)
#     print(city_dict)

# flights_long = sns.load_dataset("flights")
# print(flights_long)
#
# flights = flights_long.pivot("month", "year", "passengers")
#
# print(flights)

# list_2 = []
# list1 = [1, 2, 3]
# list2 = [4, 5, 6]
# list3 = [7, 8, 9]
#
# list_2.append(list1)
# list_2.append(list2)
# list_2.append(list3)
#
# mat = np.array(list_2)
#
# print(mat)
#
# print(mat.shape)
#
# user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')
# numerical_df = user_info_df[numericalFeatures]
# numerical_df.fillna(-1, inplace = True)
#
# corr_mat = numerical_df.corr()
#
# print(corr_mat)
#
# print(corr_mat.index)

#user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')
# numerical_df = user_info_df[numericalFeatures + ['user_id']]
#
# null_df = numerical_df[numerical_df['zhima_score'].isnull()]
#
# notnull_df = numerical_df[numerical_df['zhima_score'].notnull()]
#
# print(notnull_df)

# joblevel_df = user_info_df[['job_level', 'loan_status']]
# joblevel_df.fillna('未知', inplace = True)
# jobs = set(joblevel_df['job_level'])
# print(jobs)


# np.random.seed(2)  #设置随机种子
# df = pd.DataFrame(np.random.rand(5,4),
#                   columns=['A', 'B', 'C', 'D'])
#
# plt.boxplot(x=df.values,labels=df.columns,whis=1.5)
#
# plt.show()

# def list_generator(mean, dis, number):  # 封装一下这个函数，用来后面生成数据
#     return np.random.normal(mean, dis * dis, number)  # normal分布，输入的参数是均值、标准差以及生成的数量


# # 我们生成四组数据用来做实验，数据量分别为70-100
# list1 = list_generator(0.8531, 0.0956, 70)
# list2 = list_generator(0.8631, 0.0656, 80)
# list3 = list_generator(0.8731, 0.1056, 90)
# list4 = list_generator(0.8831, 0.0756, 100)
# s1 = pd.Series(np.array(list1))
# s2 = pd.Series(np.array(list2))
# s3 = pd.Series(np.array(list3))
# s4 = pd.Series(np.array(list4))
# # 把四个list导入到pandas的数据结构中，dataframe
# data = pd.DataFrame({"1": s1, "2": s2, "3": s3, "4": s4})
# data.boxplot()  # 这里，pandas自己有处理的过程，很方便哦。
# plt.ylabel("ylabel")
# plt.xlabel("xlabel")  # 我们设置横纵坐标的标题。
# plt.show()

# province_dict_df = pd.read_excel(ROOT_DIR + 'settings/province_badrate_classification.xlsx', encoding='utf-8')
# y = 'loan_status'
# x = set(list(province_dict_df)).difference(set([y]))
# print(x)
# #dict = province_dict_df.set_index('identity_province').T.to_dict('int')['classification']
#
# print(set(list(province_dict_df)))


condition_list = ['var == 0', 'var >=1 and var <= 2', 'var == 3']
s = 0
x = '2'
for condition in condition_list:
    condition = condition.replace("var", str(x))
    print(condition)
    if eval(condition):
        print('matched !')
        print(s)
    else:
        s += 1

dicts = {'job_level': {'missing': 0.00014964459408903854, '一般工人': 0.0032921810699588477, '一般科员': 0.0017209128320239432, '专业技术人员': 0.016610549943883276, '专员': 0.00014964459408903854, '主任/主管/组长/初级管理': 0.2680882903105125, '保安': 0.0006734006734006734, '其他': 0.00202020202020202, '军人': 0.00014964459408903854, '农林牧渔养殖人员': 0.00014964459408903854, '助理': 0.00014964459408903854, '医护人员': 0.0014964459408903852, '厨师': 0.0023943135054246166, '司机': 0.002244668911335578, '学生': 0.048484848484848485, '实习生': 0.0010475121586232697, '工程师': 0.002543958099513655, '总监/总经理/高管': 0.029405162738496072, '技术工人': 0.012719790497568275, '教师': 0.001421623643845866, '普通员工': 0.39206883651328095, '服务员': 0.001421623643845866, '法人/老板': 0.00935278713056491, '演员/运动员': 7.482229704451927e-05, '科级以上': 0.0003741114852225963, '经理/中级管理': 0.17949869060980173, '行政/人力资源': 0.005462027684249906, '设计师': 0.0008230452674897119, '财会人员': 0.00404040404040404, '销售人员': 0.011971567527123082}}
print()

# condition_list = ['var == 0', 'var >=1 and var <= 2', 'var == 3']
# s = 0
# x = '2'
# for condition in condition_list:
#     condition = condition.replace("var", str(x))
#     print(condition)
#     if eval(condition):
#         print('matched !')
#         print(s)
#     else:
#         s += 1

WOE_IV_dict = {}
WOE_IV_dict1 = {}
var = 'zhima_score_classification'
new_var = var + '_WOE'
train_data = pd.read_excel(ROOT_DIR + 'transformed_train.xlsx', encoding = 'utf-8')

merged_dict = MergeBad0(train_data, var, LABEL, direction='bad')
train_data[var] = train_data[var].map(lambda x : merged_dict[x])
print(merged_dict)
print(set(train_data[var]))

WOE_IV_dict[new_var] = CalcWOE(train_data, var, LABEL)
train_data[new_var] = train_data[var].map(lambda x : WOE_IV_dict[new_var]['WOE'][x])

print(WOE_IV_dict[new_var])

print('****************************************')

merged_dict1 = MergeBad0(train_data, var, LABEL, direction='good')
train_data[var] = train_data[var].map(lambda x : merged_dict1[x])
print(merged_dict1)
print(set(train_data[var]))

WOE_IV_dict1[new_var] = CalcWOE(train_data, var, LABEL)
train_data[new_var] = train_data[var].map(lambda x : WOE_IV_dict1[new_var]['WOE'][x])

print(WOE_IV_dict1[new_var])

