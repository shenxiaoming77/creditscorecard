#coding=utf-8

import  pandas as pd
import  numpy as np


from  featureEngineering.featureTransform.derivation import *
from featureEngineering.featureTransform.single_transform import *
from  dataPreProcess.data_split import train_test_split_func

class FeatureTransformRunner():
    def __init__(self):
        self.derivation_df = pd.DataFrame()

    def loadData(self, file):
        self.data_df = pd.read_excel(file, encoding='utf-8')

    def setData(self, df):
        self.data_df = df

    def featureDerivation(self):
        print('feature derivation:')
        register_apply_date_interval(self.data_df, self.derivation_df)
        seemingly_abnormity_application(self.data_df, self.derivation_df)

        identity_city_classification(self.data_df, self.derivation_df)
        phone_city_classification(self.data_df, self.derivation_df)

        identity_province_classification(self.data_df, self.derivation_df)

        user_age_classification(self.data_df, self.derivation_df)
        zhima_score_classification(self.data_df, self.derivation_df)
        br_score_classification(self.data_df, self.derivation_df)



    def singleFeatureTransform(self):
        job_level_combine(self.data_df)
        network_len_combine(self.data_df)

    def saveData(self, file1, file2):

        self.data_df.to_excel(ROOT_DIR + file1, index=None, encoding='utf-8')
        self.derivation_df.to_excel(ROOT_DIR + file2, index=None, encoding='utf-8')

        with open('derivation_features', 'w') as file:
            for var in [x for x in self.derivation_df.columns if x not in ['user_id']]:
                file.write(var + '\n')
        file.close()

    def mergeData(self, file1, file2):

        #self.saveData(file1, file2)

        merge_df = pd.concat([self.data_df, self.derivation_df], axis=1)
        merge_df.drop(['bill_id','phone','name','identity'], axis = 1, inplace = True)
        merge_df.to_excel(ROOT_DIR + 'transformed_train.xlsx', index = None)

    def splitData(self, file1, file2):

        #self.saveData(file1, file2)

        merge_df = pd.concat([self.data_df, self.derivation_df], axis=1)
        featurs = [x for x in merge_df.columns if x not in ['bill_id','phone','name','identity']]
        train_x, test_x, train_y, test_y = train_test_split_func(merge_df, featurs, 0.25)
        train_df = pd.concat([train_x, train_y], axis=1)
        test_df = pd.concat([test_x, test_y], axis=1)
        train_df.to_excel(ROOT_DIR + 'transformed_train.xlsx', index = None)
        test_df.to_excel(ROOT_DIR + 'transformed_test.xlsx', index = None)

if __name__ == '__main__':
    data_df = pd.read_excel(ROOT_DIR + 'preProcessed_user_info.xlsx', encoding='utf-8')

    transformRunner = FeatureTransformRunner()
    transformRunner.setData(data_df)
    transformRunner.singleFeatureTransform()
    transformRunner.featureDerivation()

    transformRunner.splitData('transformed_originFeatures_data.xlsx', 'derivation_features_data.xlsx')

