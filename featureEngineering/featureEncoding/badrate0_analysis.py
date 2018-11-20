#encoding=utf-8
import  pandas as pd
import  numpy as np

from  settings import *
from  util.scorecard_functions import existing_badrate0
from  util.scorecard_functions import MergeBad0
from  util.scorecard_functions import BinBadRate

not_monotone = ['auth_level','network_len','identity_city_classification',
                        'phone_city_classification','br_score_classification',
                        'user_age_classification',]

categoricalFeatures = FEATURE_DICT['categoricalFeatures']
numericalFeatures = FEATURE_DICT['numericalFeatures']

df = pd.read_excel(ROOT_DIR + 'transformed_train.xlsx')


regroup = BinBadRate(df, 'auth_level', LABEL)[1]
regroup_bad = regroup.sort_values(by  = 'bad_rate')
regroup_good = regroup.sort_values(by='bad_rate',ascending=False)

print(regroup_bad)
print('****************')
print(MergeBad0(df, 'auth_level', LABEL, 'bad'))
print('************************************************')
print(regroup_good)
print('****************')
print(MergeBad0(df, 'auth_level', LABEL, 'good'))