# dicts = {'a':'ff','b':'bb','1':'11','a':'aa',}
# print(list(dicts.keys()))
# print(list(dicts.values()))

from  settings import  *
import  pickle
import  pandas as pd

df1 = pd.read_excel(ROOT_DIR + 'settings/1.xls', encoding='gbk')


df2 = pd.read_excel(ROOT_DIR + 'settings/2.xls', encoding='gbk')


df3 = pd.read_excel(ROOT_DIR + 'settings/3.xls', encoding='gbk')


df4 = pd.read_excel(ROOT_DIR + 'settings/4.xls', encoding='gbk')


df5 = pd.read_excel(ROOT_DIR + 'settings/5.xls', encoding='gbk')

df6 = pd.read_excel(ROOT_DIR + 'settings/6.xls', encoding='gbk')

city_dict = {}
city_dict['一线'] = list(df1['一线'])
city_dict['新一线'] = list(df2['新一线'])
city_dict['二线'] = list(df3['二线'])
city_dict['三线'] = list(df4['三线'])
city_dict['四线'] = list(df5['四线'])
city_dict['五线'] = list(df6['五线'])

print(city_dict)

with open(ROOT_DIR + 'settings/city_classification.pkl', 'wb') as file:
    pickle.dump(city_dict, file)

with open(ROOT_DIR + 'settings/city_classification.pkl', 'rb') as file:
    city_dict = pickle.load(file)
    print(city_dict)
