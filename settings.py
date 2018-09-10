#coding=utf-8

ROOT_DIR = 'D:/conf_test/creditscored/'
FE_DIR = ROOT_DIR + 'featureEngineering/'  #featureEngineering_dir


NUM_BINS = 5


numericalFeatures = ['user_age','zhima_score','auth_level','company_college_length','br_score',
                'm1_verif_count','m1_register_count','m1_apply_request_count','m1_apply_reject_count',
                'm1_loan_offer_count','m1_repay_fail_count','m1_overdue_count','m1_repay_remind_count',
                'm3_verif_count','m3_register_count','m3_apply_request_count','m3_apply_reject_count',
                'm3_loan_offer_count','m3_repay_fail_count','m3_overdue_count','m3_repay_remind_count',
                'm6_verif_count','m6_register_count','m6_apply_request_count','m6_apply_reject_count',
                'm6_loan_offer_count','m6_repay_fail_count','m6_overdue_count','m6_repay_remind_count',
                'm12_verif_count','m12_register_count','m12_apply_request_count','m12_apply_reject_count',
                'm12_loan_offer_count','m12_repay_fail_count','m12_overdue_count','m12_repay_remind_count'
                ]
#类别型变量
categoricalFeatures = ['gender','job_level','phone_province','phone_city','identity_province','identity_city',
                'is_identical_phone_identity_province',
                'occupation','registerDate_xunYue','applyDate_xunYue','registerDate_hour','applyDate_hour',
                'network_len'
                ]

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




