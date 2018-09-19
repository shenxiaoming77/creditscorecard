#coding=utf-8

import pandas as pd
import  numpy as np
import  xgboost as xgb
from  settings import  *


from  dataPreProcess.data_split import *
from dataPreProcess.null_process import *

class PreProcessRunner():
    def setData(self, df):
        self.data_df = df

    def loadData(self, file):
        self.data_df = pd.read_excel(file, encoding='utf-8')

    def nullValueProcess(self):
        print('zhima score missing value process:')
        zhimaScore_df = pd.read_excel(ROOT_DIR + 'zhima_missingValue.xlsx')
        zhimaScore_missingValue_ByCluster(self.data_df, zhimaScore_df)
        print('job level missing value process:')
        job_level_missingValue(self.data_df)
        print('phone city missing value process:')
        phone_city_missingValue(self.data_df)
        print('phone province missing value process:')
        phone_province_missingValue(self.data_df)
        print('identity city missing value process:')
        identity_city_missingValue(self.data_df)
        print('identity province missing value process:')
        identity_province_missingValue(self.data_df)
        print('br score missing value process:')
        br_score_missingValue(self.data_df)
        print('occupation missing value process:')
        occupation_missingValue(self.data_df)
        print('network len missing value process:')
        network_len_missingValue(self.data_df)
        print('company college length missing value process:')
        company_college_length_missingValue(self.data_df)

    def splitData(self):
        featurs = [x for x in self.data_df.columns if x not in ['bill_id','phone','name','identity']]
        train_x, test_x, train_y, test_y = train_test_split(self.data_df, featurs)

    def saveData(self, file):
        print(file)
        self.data_df.to_excel(ROOT_DIR + file, index = None, encoding = 'utf-8')

if __name__ == '__main__':
    data_df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding='utf-8')
    processRunner = PreProcessRunner()
    processRunner.setData(data_df)
    processRunner.nullValueProcess()
    processRunner.saveData('preProcessed_user_info.xlsx')