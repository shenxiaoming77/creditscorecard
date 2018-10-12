#encode=utf-8

import  pandas as pd
import  numpy as np
import  xgboost as xgb
import  operator

from  settings import  *

# 针对类别变量，通过笛卡尔积得到交叉变量



features = ['gender','job_level',
            'occupation','registerDate_xunYue','applyDate_xunYue',
            'auth_level','network_len','identity_city_classification',
            'phone_city_classification','zhima_score_classification',
            'br_score_classification','user_age_classification',
            'identity_province_classification',]


def feature_cross(df):
    duplicates = []
    cross_features = []
    for x in features:
        for y in features:
            new_var = x + '_cross_' + y
            if x == y or new_var in duplicates:
                continue
            else:
                print(new_var)
                df[new_var] = df.apply(lambda row : str(row[x]) + '_' + str(row[y]), axis = 1)
                duplicates.append(y + '_cross_' +x)
                cross_features.append(new_var)

    cross_df = df[cross_features + ['loan_status', 'user_id']]

    cross_df.to_excel(ROOT_DIR + 'cross_df.xlsx', index = None)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def feature_importance(df):
    features = [x for x in df.columns if x not in ['user_id', 'loan_status']]
    for var in features:
        df[var] = pd.factorize(df[var].values, sort=True)[0]


    X = df[features]
    y = df['loan_status']

    dtrain = xgb.DMatrix(X, label=y)

    clf = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=20)

    create_feature_map(features)
    importance = clf.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key = operator.itemgetter(1), reverse = True)

    df = pd.DataFrame(importance, columns=['feature', 'score'])
    print(df)
    df.to_excel(FE_DIR + 'featurescore/cross_feature_score_single.xlsx', index=None)

if __name__ == '__main__':
    # user_info_df = pd.read_excel(ROOT_DIR + 'transformed_train_test.xlsx',
    #                              encoding='utf-8')[features + ['loan_status', 'user_id']]
    # cross_df = feature_cross(user_info_df)

    cross_df =  pd.read_excel(ROOT_DIR + 'cross_df.xlsx')

    feature_importance(cross_df)

