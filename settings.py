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

FEATURE_DICT = {
    'numericalFeatures' :
                    ['user_age',
                     'zhima_score',
                     'company_college_length',
                     'registerDate_hour',
                     'applyDate_hour',
                     'm1_verif_count',
                     'm1_register_count',
                     'm1_apply_request_count',
                     'm1_apply_reject_count',
                     'm1_loan_offer_count',
                     'm1_repay_fail_count',
                     'm1_overdue_count',
                     'm1_repay_remind_count',
                     'm3_verif_count',
                     'm3_register_count',
                     'm3_apply_request_count',
                     'm3_apply_reject_count',
                     'm3_loan_offer_count',
                     'm3_repay_fail_count',
                     'm3_overdue_count',
                     'm3_repay_remind_count',
                     'm6_verif_count',
                     'm6_register_count',
                     'm6_apply_request_count',
                     'm6_apply_reject_count',
                     'm6_loan_offer_count',
                     'm6_repay_fail_count',
                     'm6_overdue_count',
                     'm6_repay_remind_count',
                     'm12_verif_count',
                     'm12_register_count',
                     'm12_apply_request_count',
                     'm12_apply_reject_count',
                     'm12_loan_offer_count',
                     'm12_repay_fail_count',
                     'm12_overdue_count',
                     'm12_repay_remind_count'
                ],
#类别型变量
'categoricalFeatures' :
                    ['gender',
                       'job_level',
                       'phone_province',
                       'identity_province',
                       'occupation',
                       'registerDate_xunYue',
                       'applyDate_xunYue',
                       'auth_level',
                       'network_len',
                       'identity_city_classification',
                       'phone_city_classification',
                       'zhima_score_classification',
                       'br_score_classification',
                       'user_age_classification',
                       'identity_province_classification'
                    ],

'toRemoveFeatures' :
                [
                    'br_score',
                    'phone_city',
                    'identity_city',
                    'is_identical_phone_identity_province',
                    'register_date',
                    'apply_date'
                ]
}

LABEL_ENCODE_DICT = {
    'network_len' : {'(6,12]' : 2,
                     '(0,6]':1,
                     '(24,+)':4,
                     '(12,24]':3,
                     'missing':0},
    'identity_city_classification' :{ '五线':6,
                                         '二线':3,
                                         '新一线':2,
                                         '三线':4,
                                         '其他': 0,
                                         '一线':1,
                                         '四线':5
                                        },
    'phone_city_classification' : {'五线':6,
                                      '二线':3,
                                      '新一线':2,
                                      '三线':4,
                                      '其他': 0,
                                      '一线':1,
                                      '四线':5
                                     },
    'registerDate_xunYue' : {'missing':0,
                                      '上旬':1,
                                      '中旬':2,
                                      '下旬':3,
                                     },
    'applyDate_xunYue' : {'missing':0,
                                      '上旬':1,
                                      '中旬':2,
                                      '下旬':3,
                                     }
}

