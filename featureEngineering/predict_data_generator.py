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
            print(self.bin_dict)

        with open(ROOT_DIR + 'featureEngineering/featuresInModel.pkl', 'rb') as f4:
            self.featuresInModel = pickle.load(f4)

        with open(ROOT_DIR + 'featureEngineering/WOE_IV_dict.pkl', 'rb') as f5:
            self.WOE_IV_dict = pickle.load(f5)

        with open(ROOT_DIR + 'featureEngineering/badrate0_merged_dict.pkl', 'rb') as f6:
            self.badrate0_merged_dict = pickle.load(f6)

        with open(ROOT_DIR + 'featureEngineering/goodrate0_merged_dict.pkl', 'rb') as f7:
            self.goodrate0_merged_dict = pickle.load(f7)


    def data_generate(self, predict_df):

        #1. 先生成所需要的衍生特征的原始数据
        # predict_df['maxDelqL1M'] = predict_df.apply(lambda x: self.delqFeatureExtractor.DelqFeatures(x,1,'max delq'),axis=1)
        # predict_df['maxDelqL3M'] = predict_df.apply(lambda x: self.delqFeatureExtractor.DelqFeatures(x,3,'max delq'),axis=1)
        #
        # predict_df['M0FreqL3M'] = predict_df.apply(lambda x: self.delqFeatureExtractor.DelqFeatures(x,3,'M0 times'),axis=1)
        # predict_df['M1FreqL6M'] = predict_df.apply(lambda x: self.delqFeatureExtractor.DelqFeatures(x, 6, 'M1 times'), axis=1)
        # predict_df['M2FreqL3M'] = predict_df.apply(lambda x: self.delqFeatureExtractor.DelqFeatures(x, 3, 'M2 times'), axis=1)
        #
        # predict_df['avgUrateL1M'] = predict_df.apply(lambda x: self.urateFeaturesExtractor.UrateFeatures(x,1, 'mean utilization rate'),axis=1)
        # predict_df['avgUrateL3M'] = predict_df.apply(lambda x: self.urateFeaturesExtractor.UrateFeatures(x,3, 'mean utilization rate'),axis=1)
        #
        # predict_df['increaseUrateL6M'] = predict_df.apply(lambda x: self.urateFeaturesExtractor.UrateFeatures(x, 6, 'increase utilization rate'),axis=1)
        #
        # #2. categoricalFeatures下的部分特征还需要进一步合并处理
        # # M0FreqL3M不需要参与bin 分箱，可以直接进行
        # predict_df['M2FreqL3M_Bin'] = predict_df['M2FreqL3M'].apply(lambda x: int(x>=1))
        # predict_df['maxDelqL1M_Bin'] = predict_df['maxDelqL1M'].apply(lambda x: MergeByCondition(x,['==0','==1','>=2']))
        # predict_df['maxDelqL3M_Bin'] = predict_df['maxDelqL3M'].apply(lambda x: MergeByCondition(x,['==0','==1','>=2']))
        #
        # #2. 对于衍生特征进行分箱，用于后续WOE编码
        # #2.1对于连续特征： 根据bin_dict来映射,保证能够满足badrate单调
        # numericalFeatures_bin = []
        # categoricalFeatures_bin = []
        #
        # modelFeatures = [i.replace('_Bin','').replace('_WOE','') for i in self.featuresInModel]
        # for var in [f for f in self.numericalFeatures if f in modelFeatures]:
        #     newBin = var + "_Bin"
        #     print(newBin)
        #     #bin = [i.values() for i in self.bin_dict if var in i][0][0]
        #     bin = [i[var] for i in self.bin_dict if var in i][0]
        #     predict_df[newBin] = predict_df[var].apply(lambda x: AssignBin(x, bin))
        #     numericalFeatures_bin.append(newBin)
        #
        # #2.2 对于 categoricalFeatures下的部分特征 手动进行分箱合并
        # # M0FreqL3M不需要手动参与bin分箱
        #
        # predict_df['M2FreqL3M_Bin'] = predict_df['M2FreqL3M'].apply(lambda x: int(x>=1))
        # predict_df['maxDelqL1M_Bin'] = predict_df['maxDelqL1M'].apply(lambda x: MergeByCondition(x,['==0','==1','>=2']))
        # predict_df['maxDelqL3M_Bin'] = predict_df['maxDelqL3M'].apply(lambda x: MergeByCondition(x,['==0','==1','>=2']))
        # categoricalFeatures_bin = ['M2FreqL3M_Bin','maxDelqL1M_Bin','maxDelqL3M_Bin','M0FreqL3M']
        #
        # #进行分箱后的特征名称集合
        # finalFeatures_bin = numericalFeatures_bin + categoricalFeatures_bin
        #
        # #3. 对于finFeatures_bin中的特征进行WOE编码
        # for var in finalFeatures_bin:
        #     var2 = var + "_WOE"
        #     predict_df[var2] = predict_df[var].apply(lambda x: self.WOE_IV_dict[var2]['WOE'][x])
        #
        # return  predict_df[self.featuresInModel]

    def compute_woe(self, df, var):
        new_var = var + '_WOE'
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
            df[var] = df[var].map(lambda x : merged_dict[x])

        for key, value in self.goodrate0_merged_dict.items():
            var = key
            merged_dict = value
            df[var] = df[var].map(lambda x : merged_dict[x])

        #3.最后对于类别变量进行woe编码计算
        for var in self.categoricalFeatures:
            if var not in not_monotone:
                self.compute_woe(df, var)


    def numerical_feature_encoding(self, df):
        print('numerical feature encoding:')
        #对于连续变量，参照bin_dict 对于每个连续变量进行bin划分后 进行woe编码计算
        modelFeatures = [i.replace('_Bin','').replace('_WOE','') for i in self.featuresInModel]
        for var in [f for f in self.numericalFeatures if f in modelFeatures]:
            newBin = var + "_Bin"
            print(newBin)
            #bin = [i.values() for i in self.bin_dict if var in i][0][0]
            bin = [i[var] for i in self.bin_dict if var in i][0]
            df[newBin] = df[var].apply(lambda x: AssignBin(x, bin))

            self.compute_woe(df, newBin)

    def feature_transform(self, df):
        print('transform features:')

    def woe_encoding(self, df):

        self.categorical_feature_encoding()

        self.numerical_feature_encoding()
