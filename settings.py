#coding=utf-8

ROOT_DIR = 'D:/conf_test/creditscored/'
FE_DIR = ROOT_DIR + 'featureEngineering/'  #featureEngineering_dir


NUM_BINS = 5

LABEL = 'loan_status'


xgb_params = {
        'min_child_wight' : 100,
        'eta' : 0.02,
        'colsample_bytree' : 0.8,
        'max_depth' : 6,
        'subsample' : 0.8,
        'alpha' : 1,
        'gamma' : 1,
        'slient' : 1,
        'verbose_eval' : True,
        'seed' : 1024
    }







