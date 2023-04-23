#%%
import pandas as pd
import numpy as np

import useful_functions as uf


def drop_high_columns(df_raw: pd.DataFrame, to_calc_corr ,threshold=0.8):
    """
    :return 
    :param df_raw: to stat the number of NaN values
    :param threshold: threshold to distinguish what is left what to be droed
    :rtype: 
    """
    corr_matrix = to_calc_corr.corr()
    rows,cols = np.where(corr_matrix>0.8)
    get_rid_2=list()
    unique_set = set()
    for i in range(len(rows)):
        posi= corr_matrix.iloc[rows[i],cols[i]]
    # rows[i],cols[i] is the position of the element in the matrix
    # posi is the correlation value of rows[i] and cols[i]
        if posi<0.9990: # type: ignore
            col_set = {corr_matrix.index[rows[i]], corr_matrix.columns[cols[i]]}
            # this is the actual column name
            if col_set not in get_rid_2: 
                # because the set disregard the order,
                # so we need to check if the set is already in the list,
                # the pair with the same two elements
                # but different order will be regarded as same set,
                get_rid_2.append(col_set) #----> a list of set
                unique_set.add(frozenset(col_set)) #----> a set of set
    drop_col_2 = list(set([item for s in get_rid_2 for item in list(s)]))
    # drop_col_2  = list(itertools.chain(*get_rid_2))
    rank_df = df_raw.loc[:, drop_col_2].isna().sum()

    #---> index:column name, value: number of missing value
    # rank_df.sort_values(0, ascending=True)

    for i in range(len(get_rid_2)):# loop in length of the four element list
        ind = get_rid_2[i]
        for col_j in ind: # col_j is the column name
            rank_df.loc[col_j,'pair']=i # add a column of pair mark to rank_df
            rank_df.loc[col_j,'col_name']=col_j # add a column of column name to rank_df
            rank_df.loc[col_j, 'abbreviation']=col_j[0:3]
    # ndarray=rank_df.groupby('pair')[0].nlargest(1)
    return rank_df.sort_values('pair')
#%%


if __name__ == '__main__':
    path_fill = '/workspaces/CMB_match/data/2022/filled_trian.csv'
    df_train = pd.read_csv('data/2022/train.csv', index_col=0).head(500)
    df_train_filled = pd.read_csv(path_fill, index_col=0).head(500)
    df_train_filled.loc[:,'SHH_BCK']=df_train_filled.SHH_BCK.astype(str) # it was floated
    string_col =uf.dynamic_string_col(df_train_filled)

    opt_col = string_col.copy()

    df_str=pd.DataFrame()
    if 'MON_12_CUST_CNT_PTY_ID' in string_col:
        # This a Binary Y/N, the training sample has Y only, do not fill with the mode
        opt_col.remove('MON_12_CUST_CNT_PTY_ID')
        df_str.loc[:,'MON_12_CUST_CNT_PTY_ID'] = df_train_filled['MON_12_CUST_CNT_PTY_ID']
        # Direct copy
    # fill the missing value with mode
    df_str.loc[:,opt_col] = uf.mode_nan_string(df_train_filled.loc[:, opt_col])
    ## One-hot encoding
    
    dummies = pd.get_dummies(df_str,dummy_na=True).astype(int)
    #%% df numerical +df str + first two columns
    df_numercial = uf.get_numerical_df(df_train_filled)
    res = pd.concat([df_train_filled.iloc[:,:2], dummies, df_numercial],axis=1)
    print(res)
    #%% dfstr + others
    dropcol = df_str.columns
    df_train_filled.drop(dropcol, axis=1, inplace=True)
    df_train_filled = pd.concat([df_train_filled, dummies], axis=1)
    print(df_train_filled)