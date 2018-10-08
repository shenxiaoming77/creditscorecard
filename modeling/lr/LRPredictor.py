import  pandas as pd
from  sklearn.externals import  joblib
from  sklearn.metrics import  roc_auc_score
from  settings import  *

from  util.scorecard_functions import KS
from  modeling.predict_data_generator import PredictionDataGenerator


class LogisticRegressionPredictor:

    def __init__(self, platform):
        self.platform = platform
        self.model = self.load_model(platform)
        print('load local model file, platfrom:', platform)

    def load_model(self, platform):
        model_name = 'LR-Model-' + platform + '.m'
        model = joblib.load(model_name)
        return  model

    def ks_auc_eval(self, result_df):

        ks = KS(result_df, 'pred', LABEL)
        auc = roc_auc_score(result_df[LABEL], result_df['pred'])
        result = {}
        result['ks'] = ks
        result['auc'] = auc
        return  result


    def predict(self, predict_df):

        X = predict_df
        print(X)

        result_df = pd.DataFrame()

        X['intercept'] = [1] * X.shape[0]
        probas = self.model.predict_proba(X)[:, 1]
        result_df['pred'] = probas

        return  result_df



if __name__ == '__main__':

    predictor = LogisticRegressionPredictor('sklearn')
    test_df = pd.read_excel(FE_DIR + 'test_WOE_data.xlsx')
    label = test_df[LABEL]
    predict_df = test_df.drop(['user_id', LABEL], axis = 1)

    result_df = predictor.predict(predict_df)
    result_df[LABEL] = list(label)

    ks_auc = predictor.ks_auc_eval(result_df)
    print(ks_auc)

    # BasePoint, PDO = 500,50
    # result_df['score'] = result_df['pred'].apply(lambda x: Prob2Score(x, BasePoint, PDO))
    # print(result_df)
    # plt.hist(result_df['score'],bins=100)
    # plt.show()

