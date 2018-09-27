from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy as np
import  pandas as pd
import pickle

from  settings import  *


class  CorrelationAnalysisSelection:

    def set_data(self, df):
        self.data_df = df

    #衍生特征之间的相关性计算
    def derived_feature_correlation_analysis(self, derivedFeature_data):

        feature_list = [x for x in derivedFeature_data.columns if x not in ['user_id']]
        print(feature_list)

        for x in feature_list:
            for y in feature_list:
                if x != y:
                    x_data = derivedFeature_data[x].astype('int')
                    y_data = derivedFeature_data[y].astype('int')
                    print(x, '      ', y, '     ', np.corrcoef(x_data,
                                                               y_data)[0, 1])

    def visualization(self, df, features):
        x = df[features]
        f, ax = plt.subplots(figsize=(10, 8))
        corr = x.corr()   #相关系数矩阵
        sns.heatmap(corr,
                    mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True,
                    ax=ax)




    #WOE编码后进行单变量与多变量分析
    def woe_feature_analysis(self):
        with open(ROOT_DIR +  'featureEngineering/WOE_IV_dict.pkl', 'rb') as f:
            WOE_IV_dict = pickle.load(f)

        print('WOE FEATURES:')
        print(WOE_IV_dict.keys())
        return

        '''
        1.先进行单变量分析：按IV值进行排序,过滤掉IV值过低的特征
        '''
        high_IV = [(var, value['IV']) for var, value in WOE_IV_dict.items() if value['IV'] >= 0.01]
        #按IV值进行排序
        high_IV_sorted = sorted(high_IV, key = lambda tuple : tuple[1], reverse = True)

        for item in high_IV_sorted:
            print(item)

        '''
        2.进入多变量分析环节：
            特征相关性检测
            多重共线性检测
        '''
        #2.1 先进行特征之间的两两相关性检测分析,针对相关性大于0.7的两个特征，进行IV值重要性比较筛选
        deleted_index = []
        num_vars = len(high_IV_sorted)
        #item in high_IV_sorted: ('zhima_score_Bin_WOE', 0.28597215298299217)
        for i in range(num_vars):
            if i in deleted_index:
                continue
            x = high_IV_sorted[i][0]
            for j in range(num_vars):
                if i == j or j in deleted_index:
                    continue
                y = high_IV_sorted[j][0]
                roh = np.corrcoef(train_data[x], train_data[y])[0, 1]
                #相关性过高，进一步观察两个特征各自的IV值重要性
                if abs(roh) > 0.7:
                    print(x, '      ', y, '     ', roh)
                    x_IV = high_IV_sorted[i][1]
                    y_IV = high_IV_sorted[j][1]
                    if x_IV > y_IV:
                        deleted_index.append(j)
                    else:
                        deleted_index.append(i)

        #最终留下进行相关性分析与筛选后的特征集合
        single_analysis_vars = [high_IV_sorted[i][0] for i in range(num_vars) if i not in deleted_index]

        print(single_analysis_vars)

        #通过可视化来查看特征两两相关性情况
        self.visualization(train_data, single_analysis_vars)

        #进行多重共线性分析
        #在单变量分析基础上对single_analysis_vars来进行分析
        X = np.matrix(train_data[single_analysis_vars])

        print(X)

        VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        print (max(VIF_list))
        # 最大的VIF是 3.429，小于10，因此这一步认为没有多重共线性
        multi_analysis = single_analysis_vars

        print(multi_analysis)
        with open(ROOT_DIR + 'featureEngineering/multi_analysis_feature_list.pkl', 'wb') as f:
            pickle.dump(multi_analysis, f)






if __name__ == '__main__':
    train_data = pd.read_excel(ROOT_DIR + 'featureEngineering/train_WOE_data.xlsx')
    analysis= CorrelationAnalysisSelection()
    analysis.set_data(train_data)
    analysis.woe_feature_analysis()


