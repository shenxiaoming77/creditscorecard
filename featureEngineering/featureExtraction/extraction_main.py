import  pandas as pd
from  settings import  *


class ExtractionMain:

    def initial(self):

        print('save to feature data file.....')


    def feature_extract(self):

        self.initial()





    def save(self, feature_data_df):
        feature_data_df.to_csv(ROOT_DIR + 'featureEngineering/train_derived_feature_data.csv', index = None)

    #其中某一类特征重新计算后，update整体训练特征集
    def update_feature(self):
        print('update feature set for training model.....')


if __name__ == '__main__':
    extractionMain = ExtractionMain()
    #feature_data_df = extractionMain.feature_extract()
    #extractionMain.update_feature()
