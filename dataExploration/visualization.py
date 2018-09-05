import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from  settings import  *

df = pd.read_excel(ROOT_DIR + 'user_info.xlsx', encoding = 'utf-8')

total = len(df['job_level'])

features_list = numericalFeatures + categoricalFeatures

def visualization_null(df, var):
    null_count = len(df[var]) - df[var].count()

    print(null_count)
    return  null_count

#所有原始特征的空值情况统计
def  null_counts_visualization():
    null_count_dict = {}

    for var in features_list:
        null_count = visualization_null(df, var)
        null_count_dict[var] = null_count

    #升序排列
    sorted_dict = sorted(null_count_dict.items(), key = lambda  d : d[1], reverse = True)
    x = []
    y = []
    i = 0
    for item in sorted_dict:
        if i > 5:
            break
        x.append(item[0])
        y.append(item[1] * 1.0 / total)
        i += 1

    x_label='features'
    y_label= 'NAN Count Num'
    title='NAN Count for Features'
    plt.bar(x, y, width = 0.35, facecolor = 'yellowgreen')
    plt.title(title)
    plt.xticks(range(len(x) + 1), x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig('./visualization_pics/null_count.png')
    plt.show()

#查看某一个特征的值分布情况
def feature_distribution_visualization(var):
    values = list(df.groupby(var)[var].count())
    keys = list(dict(list(df.groupby(var)[var])).keys())

    for i in range(len(keys)):
        key = keys[i]
        value = values[i]


    plt.show()
    print(values)


#null_counts_visualization()

feature_distribution_visualization(var = 'br_score')
