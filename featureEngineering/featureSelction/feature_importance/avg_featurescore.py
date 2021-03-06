#coding=utf-8


import pandas as pd 
import os

from  settings import  *

files = os.listdir(FE_DIR + 'featurescore')
fs = {}
for f in files:
    t = pd.read_csv(FE_DIR + 'featurescore/' + f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if fs.has_key(key):
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

with open(FE_DIR + 'featurescore/rank_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)




