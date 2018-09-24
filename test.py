# dicts = {'a':'ff','b':'bb','1':'11','a':'aa',}
# print(list(dicts.keys()))
# print(list(dicts.values()))

from  settings import  *
import  pickle
import  pandas as pd
import  numpy as np
import  seaborn as sns
import  matplotlib.pyplot as plt

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