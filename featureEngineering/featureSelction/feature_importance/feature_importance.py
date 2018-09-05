#coding=utf-8
import  pickle
import  pandas as pd
import  xgboost as xgb
from  sklearn.ensemble import  GradientBoostingClassifier
import  operator

from  settings import  *
import  os
train_data = pd.read_csv(ROOT_DIR + 'featureEngineering/train_WOE_data.csv')

if not os.path.exists(FE_DIR + 'featurescore'):
    os.mkdir(FE_DIR + 'featurescore')

def create_feature_map(features):
    outfile = open(FE_DIR + 'xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def get_fscore_xgb(train_data, features):
    X = train_data[features]
    y = train_data['label']

    dtrain = xgb.DMatrix(X, label=y)

    clf = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=20
    )

    create_feature_map(features)
    importance = clf.get_fscore(fmap=FE_DIR + 'xgb.fmap')
    importance = sorted(importance.items(), key = operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'score'])
    print(df)
    df.to_csv(FE_DIR + 'featurescore/feature_score_single.csv', index=None)


def get_fscore_gbdt(train_data, features):
    X = train_data[features]
    y = train_data['label']

    gbClassifier = GradientBoostingClassifier()
    model = gbClassifier.fit(X, y)
    importance = model.feature_importances_.tolist()

    featuresImportance = zip(features, importance)
    featuresImportanceSorted = sorted(featuresImportance, key = lambda k : k[1], reverse = True)
    print(featuresImportanceSorted)

    return  featuresImportanceSorted



if __name__ == '__main__':
    with open(ROOT_DIR + 'featureEngineering/multi_analysis_feature_list.pkl', 'rb') as f:
        features = pickle.load(f)

    get_fscore_xgb(features)
