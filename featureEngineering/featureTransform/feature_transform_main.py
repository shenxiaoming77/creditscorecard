#coding=utf-8

import  pandas as pd
import  numpy as np


from  featureEngineering.featureTransform.derivation import *
from featureEngineering.featureTransform.single_transform import *

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

    def saveData(self, file):
        self.data_df.to_excel(ROOT_DIR + file, index=None, encoding='utf-8')
        self.derivation_df.to_excel(ROOT_DIR + 'derivation_features_data.xlsx', index=None)

        with open('derivation_features', 'w') as file:
            for var in [x for x in self.derivation_df.columns if x not in ['user_id']]:
                file.write(var + '\n')
        file.close()

if __name__ == '__main__':
    data_df = pd.read_excel(ROOT_DIR + 'preProcessed_user_info.xlsx', encoding='utf-8')

    transformRunner = FeatureTransformRunner()
    transformRunner.setData(data_df)

    transformRunner.singleFeatureTransform()
    transformRunner.featureDerivation()

    transformRunner.saveData('transformed_user_info.xlsx')

