# dicts = {'a':'ff','b':'bb','1':'11','a':'aa',}
# print(list(dicts.keys()))
# print(list(dicts.values()))

from  settings import  *
import  pickle
import  pandas as pd
import  numpy as np
import  seaborn as sns

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

user_info_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')
# numerical_df = user_info_df[numericalFeatures + ['user_id']]
#
# null_df = numerical_df[numerical_df['zhima_score'].isnull()]
#
# notnull_df = numerical_df[numerical_df['zhima_score'].notnull()]
#
# print(notnull_df)

joblevel_df = user_info_df[['job_level', 'loan_status']]
joblevel_df.fillna('未知', inplace = True)
jobs = set(joblevel_df['job_level'])
print(jobs)
