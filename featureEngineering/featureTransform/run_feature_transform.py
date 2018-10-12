#coding=utf-8

import  pandas as pd
import  numpy as np


from  featureEngineering.featureTransform.feature_derivation import *
from featureEngineering.featureTransform.single_transform import *
from  dataPreProcess.data_split import train_test_split_func

class FeatureTransformRunner():
    def __init__(self):
        self.derivation_df = pd.DataFrame()

        self.derivation_features = []

    def loadData(self, file):
        self.data_df = pd.read_excel(file, encoding='utf-8')
        self.derivation_df['user_id'] = self.data_df['user_id']

    def setData(self, df):
        self.data_df = df
        self.derivation_df['user_id'] = self.data_df['user_id']

    #单变量衍生与交叉特征衍生
    def featureDerivation(self):
        print('feature derivation:')
        register_apply_date_diff(self.data_df, self.derivation_df)
        seemingly_abnormity_application(self.data_df, self.derivation_df)

        identity_city_classification(self.data_df, self.derivation_df)
        feature_labelEncode(self.derivation_df, 'identity_city_classification')

        phone_city_classification(self.data_df, self.derivation_df)
        feature_labelEncode(self.derivation_df, 'phone_city_classification')

        identity_province_classification(self.data_df, self.derivation_df)

        user_age_classification(self.data_df, self.derivation_df)
        zhima_score_classification(self.data_df, self.derivation_df)
        br_score_classification(self.data_df, self.derivation_df)


    #单变量进行值域变换,并针对部分类别变量进行label encode
    def singleFeatureTransform(self):
        job_level_combine(self.data_df)
        network_len_combine(self.data_df)

        feature_labelEncode(self.data_df, 'applyDate_xunYue')
        feature_labelEncode(self.data_df, 'registerDate_xunYue')
        feature_labelEncode(self.data_df, 'network_len')

    def saveData(self):

        self.derivation_df.to_excel(ROOT_DIR + 'derivation_features_data.xlsx', index=None, encoding='utf-8')

        with open(ROOT_DIR + 'featureEngineering/derivation_features', 'w') as file:
            for var in [x for x in self.derivation_df.columns if x not in ['user_id']]:
                file.write(var + '\n')
        file.close()


    def generate_train_test(self):

        merge_df = pd.merge(self.data_df, self.derivation_df, on='user_id', how='left')
        merge_df.drop(['bill_id', 'phone', 'name', 'identity'], axis = 1, inplace = True)

        features = [x for x in merge_df.columns if x not in ['loan_status']]
        train_x, test_x, train_y, test_y = train_test_split_func(merge_df, features, 0.25)
        train_df = pd.concat([train_x, train_y], axis=1)
        test_df = pd.concat([test_x, test_y], axis=1)

        merge_df.to_excel(ROOT_DIR + 'transformed_train_test.xlsx', index = None)
        train_df.to_excel(ROOT_DIR + 'transformed_train.xlsx', index = None)
        test_df.to_excel(ROOT_DIR + 'transformed_test.xlsx', index = None)

    def split_data(self, df, features):
        train_x, test_x, train_y, test_y = train_test_split_func(df, features, 0.25)
        train_df = pd.concat([train_x, train_y], axis=1)
        test_df = pd.concat([test_x, test_y], axis=1)

        train_df.to_excel(ROOT_DIR + 'transformed_train.xlsx', index = None)
        test_df.to_excel(ROOT_DIR + 'transformed_test.xlsx', index = None)


if __name__ == '__main__':
    # data_df = pd.read_excel(ROOT_DIR + 'preProcessed_user_info.xlsx', encoding='utf-8')
    #
    # transformRunner = FeatureTransformRunner()
    # transformRunner.setData(data_df)
    # transformRunner.singleFeatureTransform()
    # transformRunner.featureDerivation()
    #
    # transformRunner.generate_train_test()

    transformRunner = FeatureTransformRunner()

    df1 = pd.read_excel(ROOT_DIR + 'transformed_train_test.xlsx')
    features1 = [x for x in df1.columns if x not in ['loan_status']]
    df2 = pd.read_excel(ROOT_DIR + 'cross_df.xlsx').drop(['loan_status'], axis = 1)
    features2 = [x for x in df2.columns if x not in ['user_id','loan_status']]

    merge_df = pd.merge(df1, df2, on='user_id', how='left')

    transformRunner.split_data(merge_df, features1 + features2)

