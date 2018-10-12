import  pandas as pd
import  numpy as np
import  pickle
from  settings import  *
from  util.scorecard_functions import  *

from  settings import  *
from  settings import  FEATURE_DICT

"""
对预处理后的特征进行分箱与WOE编码
"""
class WOEEncoding:

    def __init__(self, file):
        self.categoricalFeatures = FEATURE_DICT['categoricalFeatures']
        self.numericalFeatures = FEATURE_DICT['numericalFeatures']
        self.crossFeatures = pd.read_excel(FE_DIR + 'cross_features.xlsx')['feature']
        self.WOE_IV_dict = {}
        self.toRemove_list = FEATURE_DICT['toRemoveFeatures']
        self.bin_dict = []

        self.badrate0_merged_dict = {}
        self.goodrate0_merged_dict = {}

        self.loadData(file)

    def loadData(self, file):
        self.train_data = pd.read_excel(file, encoding='utf-8')

    def setData(self, data):
        self.train_data = data

    def feature_encoding_process(self):

        #类别型变量与连续数值型变量分别进行卡方分箱与WOE编码
        self.categorical_feature_encoding()

        self.numerical_feature_encoding()

        #保存最终的WOE编码结果 以及相关特征分箱等信息，用于后续多变量分析 以及模型训练
        self.save()


        '''
         badrate不单调并且需要卡方分箱合并的类别变量:
        'auth_level': == 0, == 1, == 2, >= 3
        'network_len': ==0, ==1, >=2
        'identity_city_classification': ==0, ==1, >=2 and <= 3, ==4, >=5
        'phone_city_classification':==0, >= 1 and <=2, ==3, ==4, >=5
        'br_score_classification':==0, >=1 and <= 2, ==3
        'user_age_classification':==0, ==1, >=2
        '''
    def not_monotone_feature_process(self):
        self.train_data['auth_level_Bin'] = self.train_data['auth_level']\
            .apply(lambda x: MergeByCondition(x, ['var == 0','var == 1', 'var == 2', 'var >= 3']))
        print(self.train_data.groupby('auth_level_Bin')[LABEL].mean())

        self.train_data['network_len_Bin'] = self.train_data['network_len']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var == 1', 'var >= 2']))
        print(self.train_data.groupby('network_len_Bin')[LABEL].mean())

        self.train_data['identity_city_classification_Bin'] = self.train_data['identity_city_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var == 1', 'var >= 2 and var <= 3', 'var == 4', 'var >= 5']))
        print(self.train_data.groupby('identity_city_classification_Bin')[LABEL].mean())

        self.train_data['phone_city_classification_Bin'] = self.train_data['phone_city_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var >= 1 and var <=2', 'var == 3', 'var == 4', 'var >= 5']))
        print(self.train_data.groupby('phone_city_classification_Bin')[LABEL].mean())

        self.train_data['br_score_classification_Bin'] = self.train_data['br_score_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var >= 1 and var <= 2', 'var == 3']))
        print(self.train_data.groupby('br_score_classification_Bin')[LABEL].mean())

        self.train_data['user_age_classification_Bin'] = self.train_data['user_age_classification']\
            .apply(lambda x : MergeByCondition(x, ['var == 0', 'var == 1', 'var >= 2']))
        print(self.train_data.groupby('user_age_classification_Bin')[LABEL].mean())


        #计算WOE值
        self.compute_woe('auth_level_Bin')
        self.compute_woe('network_len_Bin')
        self.compute_woe('identity_city_classification_Bin')
        self.compute_woe('phone_city_classification_Bin')
        self.compute_woe('br_score_classification_Bin')
        self.compute_woe('user_age_classification_Bin')


    def compute_woe(self, var):
        new_var = var + '_WOE'
        self.WOE_IV_dict[new_var] = CalcWOE(self.train_data, var, LABEL)
        self.train_data[new_var] = self.train_data[var].map(lambda x : self.WOE_IV_dict[new_var]['WOE'][x])


    def categorical_feature_encoding(self):

        not_monotone = ['auth_level','network_len','identity_city_classification',
                        'phone_city_classification','br_score_classification',
                        'user_age_classification',]


        #1. 针对badrate不单调的类别特征进行处理
        self.not_monotone_feature_process()

        #2. 对于其他类别变量，需要进一步检测每个bin是否存在零坏样本的情况，如果存在则需要进行merge
        for var in self.categoricalFeatures:
            if var not in not_monotone:
                if existing_badrate0(self.train_data, var, LABEL, 'bad'):
                    merged_dict = MergeBad0(self.train_data, var, LABEL, direction='bad')
                    self.train_data[var] = self.train_data[var].apply(lambda x : merged_dict[x])
                    self.badrate0_merged_dict[var] = merged_dict

        print('badrate0 merged dict:')
        print(self.badrate0_merged_dict)

        #3. 对于其他类别变量，在零坏样本检测之后，需要进行零好样本的bin检测，如果存在也需要进行merge
        for var in self.categoricalFeatures:
            if var not in not_monotone:
                if existing_badrate0(self.train_data, var, LABEL, 'good'):
                    merged_dict = MergeBad0(self.train_data, var, LABEL, direction='good')
                    self.train_data[var] = self.train_data[var].apply(lambda x : merged_dict[x])
                    self.goodrate0_merged_dict[var] = merged_dict

        print('goodrate0 merged dict:')
        print(self.goodrate0_merged_dict)

        #针对其他单调的类别型变量，检查是否有一箱的样本数量占比低于5%。 如果有，将该变量进行合并
        #根据是否存在样本数量有小于5%的bin，将特征分为small_bin_var与large_bin_var两个集合
        #依次对small_bin_var与large_bin_var两个集合中的特征进行分析处理
        small_bin_var = []
        large_bin_var = []
        N = self.train_data.shape[0]
        for var in self.categoricalFeatures:
            if var not in not_monotone:
                total = self.train_data.groupby([var])[var].count()
                pcnt = total * 1.0 / N
                if min(pcnt) < 0.05:
                    small_bin_var.append({var : pcnt.to_dict()})
                else:
                    large_bin_var.append(var)


        #针对 small_bin_var中的变量进行处理
        print('samll bin list:')
        for i in small_bin_var:
            print(i)

        for var in small_bin_var:
            for key, value in var.items():
                print(key)
                self.compute_woe(key)

        #:针对large_bin_var中的变量进行处理
        #对于不需要分箱合并，原始特征的badrate就已经单调的变量直接计算WOE和IV值
        for var in large_bin_var:
            self.compute_woe(var)


    '''
    对于数值型变量，需要先分箱，再计算WOE、IV
    分箱的结果需要满足：
    1:箱数不超过5
    2:bad rate单调
    3:每箱占比不低于5%
    '''
    def numerical_feature_encoding(self):

        for var in self.numericalFeatures + list(self.crossFeatures):
            max_bins = NUM_BINS
            print(var)
            #先进行一次卡方分箱，将特征数据离散化，并且按照卡方合并的原理，将原始特征的bin数控制在num_bin范围以内
            bin = ChiMerge(self.train_data, var, LABEL, max_interval=max_bins, minBinPcnt= 0.05)
            print('chiMerge bin: ', bin)
            new_var = var + '_Bin'
            self.train_data[new_var] = self.train_data[var].apply(lambda x : AssignBin(x, bin))

            while not BadRateMonotone(self.train_data, new_var, LABEL):
                print('not monotone, ChiMerge to make badrate monotone  for var: ', var)
                max_bins -= 1  #降低分箱数，进一步合并，再判断badrate是否能满足单调
                print('max bin:', max_bins)
                bin = ChiMerge(self.train_data, var, LABEL, max_interval=max_bins, minBinPcnt= 0.05)
                self.train_data[new_var] = self.train_data[var].apply(lambda x: AssignBin(x, bin))

            #满足单调性后计算WOE值
            self.compute_woe(new_var)
            self.bin_dict.append({var: bin})



    def save(self):

        #print(self.train_data.columns)
        self.train_data.to_excel(ROOT_DIR + 'featureEngineering/train_WOE_data.xlsx',
                                 index=None, encoding='utf-8')

        #将所有经过WOE编码的新特征及相关WOE,IV值保存在本地
        with open(ROOT_DIR + 'featureEngineering/WOE_IV_dict.pkl', 'wb') as f:
            print(self.WOE_IV_dict.keys())
            pickle.dump(self.WOE_IV_dict, f)

        with open(ROOT_DIR + 'featureEngineering/numericalFeatures.pkl', 'wb') as f1:
            pickle.dump(self.numericalFeatures, f1)

        with open(ROOT_DIR + 'featureEngineering/categoricalFeatures.pkl', 'wb') as f2:
            pickle.dump(self.categoricalFeatures, f2)

        with open(ROOT_DIR + 'featureEngineering/bin_dict.pkl', 'wb') as f3:
            pickle.dump(self.bin_dict, f3)

        with open(ROOT_DIR + 'featureEngineering/badrate0_merged_dict.pkl', 'wb') as f4:
            pickle.dump(self.badrate0_merged_dict, f4)

        with open(ROOT_DIR + 'featureEngineering/goodrate0_merged_dict.pkl', 'wb') as f5:
            pickle.dump(self.goodrate0_merged_dict, f5)

if __name__ == '__main__':
    woeEncoding = WOEEncoding(ROOT_DIR + 'transformed_train.xlsx')
    woeEncoding.feature_encoding_process()