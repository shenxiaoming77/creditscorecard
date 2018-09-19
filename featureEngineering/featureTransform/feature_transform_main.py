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


if __name__ == '__main__':
    data_df = pd.read_excel(ROOT_DIR + 'preProcessed_user_info.xlsx', encoding='utf-8')

    transformRunner = FeatureTransformRunner()
    transformRunner.setData(data_df)


