import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

# temp_df = pd.read_csv('../outputs/df_final.csv', sep='\t')
#
# with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
#     print(temp_df)
def data_albrecht():
    raw_data = loadarff("../testData/albrecht.arff")
    df_data = pd.DataFrame(raw_data[0])
    new_alb = df_data.drop(columns=['FPAdj', 'RawFPcounts', 'AdjFP'])
    return new_alb

def data_github_new():
    df_raw = pd.read_csv('../testData/data_2/z3_monthly.csv', sep=',')
    df_raw = df_raw.drop(columns=['dates'])

    last_col = 'monthly_commits'

    cols = list(df_raw.columns.values)
    cols.pop(cols.index(last_col))
    df_adjust = df_raw[cols+[last_col]]

    return df_adjust


# def data_github_sum():
#     df_raw = pd.read_csv('../testData/PyGithub_monthly.csv', sep=',')
#     df_raw = df_raw.drop(columns=['dates'])
#
#     last_col = 'monthly_commits'
#
#     cols = list(df_raw.columns.values)
#     cols.pop(cols.index(last_col))
#     df_adjust = df_raw[cols+[last_col]]
#
#     return df_adjust


if __name__ == '__main__':
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data_github_new())
