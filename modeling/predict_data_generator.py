import  pandas as pd
import  pickle

from settings import  *

from  util.scorecard_functions import *

class PredictionDataGenerator:

    def __init__(self):

        with open(ROOT_DIR +  'featureEngineering/numericalFeatures.pkl', 'rb') as f1:
            self.numericalFeatures = pickle.load(f1)

        with open(ROOT_DIR +  'featureEngineering/categoricalFeatures.pkl', 'rb') as f2:
            self.categoricalFeatures = pickle.load(f2)

        with open(ROOT_DIR + 'featureEngineering/bin_dict.pkl', 'rb') as f3:
            self.bin_dict = pickle.load(f3)

        with open(ROOT_DIR + 'featureEngineering/featuresInModel.pkl', 'rb') as f4:
            self.featuresInModel = pickle.load(f4)
            print('features in Model:')
            print(self.featuresInModel)

        with open(ROOT_DIR + 'featureEngineering/WOE_IV_dict.pkl', 'rb') as f5:
            self.WOE_IV_dict = pickle.load(f5)

        with open(ROOT_DIR + 'featureEngineering/badrate0_merged_dict.pkl', 'rb') as f6:
            self.badrate0_merged_dict = pickle.load(f6)

        with open(ROOT_DIR + 'featureEngineering/goodrate0_merged_dict.pkl', 'rb') as f7:
            self.goodrate0_merged_dict = pickle.load(f7)


    def compute_woe(self, df, var):
        new_var = var + '_WOE'
        print(new_var)
        df[new_var] = df[var].map(lambda x : self.WOE_IV_dict[new_var]['WOE'][x])

    def categorical_feature_encoding(self, df):
        print('categorical feature encoding:')
        not_monotone = ['auth_level', 'network_len','identity_city_classification',
                        'phone_city_classification','br_score_classification','user_age_classification']
        #1. badrate不单调的类别特征进行合并，在符合业务逻辑前提下保证badrate单调性
        df['auth_level_Bin'] = df['auth_level']\
            .apply(lambda x: MergeByCondition(x, ['var == 0','var == 1', 'var == 2', 'var >= 3']))
        df['network_len_Bin'] = df['network_len']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var == 1', 'var >= 2']))
        df['identity_city_classification_Bin'] = df['identity_city_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var == 1', 'var >= 2 and var <= 3', 'var == 4', 'var >= 5']))
        df['phone_city_classification_Bin'] = df['phone_city_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var >= 1 and var <=2', 'var == 3', 'var == 4', 'var >= 5']))
        df['br_score_classification_Bin'] = df['br_score_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var >= 1 and var <= 2', 'var == 3']))
        df['user_age_classification_Bin'] = df['user_age_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var == 1', 'var >= 2']))

        self.compute_woe(df, 'auth_level_Bin')
        self.compute_woe(df, 'network_len_Bin')
        self.compute_woe(df, 'identity_city_classification_Bin')
        self.compute_woe(df, 'phone_city_classification_Bin')
        self.compute_woe(df, 'br_score_classification_Bin')
        self.compute_woe(df, 'user_age_classification_Bin')

        #2.对于其他类别变量，需要进一步检测每个bin是否存在零坏样本或零好样本的情况，如果存在则需要进行merge
        for key, value in self.badrate0_merged_dict.items():
            var = key
            merged_dict = value
            df[var] = df[var].map(lambda x : badrate0_dict_map(x, merged_dict))

        for key, value in self.goodrate0_merged_dict.items():
            var = key
            merged_dict = value
            df[var] = df[var].map(lambda x : badrate0_dict_map(x, merged_dict))

        #3.最后对于类别变量进行woe编码计算
        for var in self.categoricalFeatures:
            if var not in not_monotone:
                self.compute_woe(df, var)


    def numerical_feature_encoding(self, df):
        crossFeatures = pd.read_excel(FE_DIR + 'cross_features.xlsx')['feature']
        print('numerical feature encoding:')
        #对于连续变量，参照预训练好的bin_dict分箱模型， 对于每个连续变量进行bin划分后 进行woe编码计算
        modelFeatures = [i.replace('_Bin','').replace('_WOE','') for i in self.featuresInModel]
        for var in [f for f in self.numericalFeatures + list(crossFeatures) if f in modelFeatures]:
            newBin = var + "_Bin"
            print(newBin)
            #bin = [i.values() for i in self.bin_dict if var in i][0][0]
            bin = [i[var] for i in self.bin_dict if var in i][0]
            df[newBin] = df[var].apply(lambda x: AssignBin(x, bin))

            self.compute_woe(df, newBin)

    def feature_transform(self, df):
        print('transform features:')

    def data_generate(self, predict_df):

        self.categorical_feature_encoding(predict_df)

        self.numerical_feature_encoding(predict_df)

        print('woe encodered features before feature selection:')
        print(predict_df.columns)

        return  predict_df[self.featuresInModel]


if __name__ == '__main__':

    generator = PredictionDataGenerator()
    data_df = pd.read_excel(ROOT_DIR + 'transformed_test.xlsx', encoding='utf-8')
    label_data = data_df[['user_id', 'loan_status']]
    train_WOE_data = generator.data_generate(data_df)

    train_WOE_data = pd.concat([label_data, train_WOE_data], axis=1)

    print(train_WOE_data.columns)

    train_WOE_data.to_excel(FE_DIR + 'test_WOE_data.xlsx', index = None)

