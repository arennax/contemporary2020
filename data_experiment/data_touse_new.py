import pandas as pd
import numpy as np
import os, time


def data_contemporary(repo_name, directory):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    df_raw = df_raw.drop(columns=['dates'])
    last_col = 'monthly_commits'
    cols = list(df_raw.columns.values)
    cols.pop(cols.index(last_col))
    df_adjust = df_raw[cols+[last_col]]

    return df_adjust


if __name__ == '__main__':

    repo_pool = []
    path = r'../data_experiment/contemporary/'
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            repo_pool.append(os.path.join(filename))
    print(len(repo_pool))

    # path = r'../data_experiment/contemporary/'
    # repo = "goby_monthly.csv"
    # print(data_contemporary(repo, path))
