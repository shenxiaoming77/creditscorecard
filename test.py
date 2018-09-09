# dicts = {'a':'ff','b':'bb','1':'11','a':'aa',}
# print(list(dicts.keys()))
# print(list(dicts.values()))

from  settings import  *
import  pickle

city_level_dict = {}
for i in city_00:
    city_level_dict[i] = '一线'

for i in city_01:
    city_level_dict[i] = '新一线'

for i in city_02:
    city_level_dict[i] = '二线'

for i in city_03:
    city_level_dict[i] = '三线'

for i in city_04:
    city_level_dict[i] = '四线'

for i in city_05:
    city_level_dict[i] = '五线'

print(city_level_dict)

with open(ROOT_DIR + 'city_level.pkl', 'wb') as file1:
    pickle.dump(city_level_dict, file1)

with open(ROOT_DIR + 'city_level.pkl', 'rb') as file2:
    city_dict = pickle.load(file2)

print(city_dict)