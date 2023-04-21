#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from ydata_profiling import ProfileReport
df_train = pd.read_csv("train.csv")
# df_testA = pd.read_csv('test_A榜.csv')
# df_testB = pd.read_csv('test_B榜.csv')
string_col = ['MON_12_CUST_CNT_PTY_ID',
'AI_STAR_SCO',
'WTHR_OPN_ONL_ICO',
'SHH_BCK',
'LGP_HLD_CARD_LVL',
'NB_CTC_HLD_IDV_AIO_CARD_SITU']
#%%
df_train.isnull().sum().sort_values(ascending=False)
_string_col = list(set(df_train.columns) & set(string_col))
    # reconfirm the columns
_df_numerical = df_train.drop(_string_col, axis=1).iloc[:, 2:].astype(float)
df_str = df_train.loc[:, _string_col]
pd.isna(_df_numerical.iloc[1,2])
#%%
print(_df_numerical.mean())