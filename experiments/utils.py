from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn import preprocessing
import pandas as pd
import numpy as np


def KFold_df(df, folds=3):
    # kf = KFold(n_splits=folds)
    # df = shuffle(df)
    #
    # for train, test in kf.split(df.index):
    #     trainData = df.iloc[train]
    #     testData = df.iloc[test]
    #     yield trainData, testData

    trainData = df.iloc[:-1]
    testData = df.iloc[-1:]
    yield trainData, testData


def normalize(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled, columns=df.columns, index=df.index)
    lst_col = df.columns[-1]
    df_normalized[lst_col] = df[lst_col]
    return df_normalized


def mre_calc(y_predict, y_actual):
    mre = []
    for predict, actual in zip(y_predict, y_actual):
        if actual == 0:
            if predict == 0:
                mre.append(0)
            elif abs(predict) <= 1:
                mre.append(1)
            else:
                mre.append(round(abs(predict - actual)+1 / (actual+1), 3))
        else:
            mre.append(round(abs(predict - actual) / (actual), 3))
    mMRE = np.median(mre)
    return mMRE


def sa_calc(Y_predict, Y_actual, X_actual):
    Absolute_Error = 0
    for predict, actual in zip(Y_predict, Y_actual):
        Absolute_Error += abs(predict - actual)
    Mean_Absolute_Error = Absolute_Error / (len(Y_predict))
    random_guess = np.mean(X_actual)
    AE_random_guess = 0
    for predict in Y_predict:
        AE_random_guess += abs(predict - random_guess)
    MAE_random_guess = AE_random_guess / (len(Y_predict))
    if MAE_random_guess == 0:
        sa_error = round((1 - (Mean_Absolute_Error+1) / (MAE_random_guess+1)), 3)
    else:
        sa_error = round((1 - Mean_Absolute_Error / MAE_random_guess), 3)
    return sa_error


def data_goal_arrange(repo_name, directory):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    df_raw = df_raw.drop(columns=['dates'])
    last_col = 'number_of_commits'
    cols = list(df_raw.columns.values)
    cols.pop(cols.index(last_col))
    df_adjust = df_raw[cols+[last_col]]

    return df_adjust




if __name__ == '__main__':

    from testData.contemp_data_touse import *

    dataset = data_github_new()

    for train, test in KFold_df(dataset, 3):

        train_input = train.iloc[:, :-1]
        train_actual_effort = train.iloc[:, -1]
        test_input = test.iloc[:, :-1]
        test_actual_effort = test.iloc[:, -1]

    print(data_github_new())
