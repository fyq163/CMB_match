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
    import os
    path = '../data/train.csv'
    print(
        os.path.dirname(os.path.dirname(__file__)+'/data/train.csv')
          )
    # '/data/train.csv'
    # print(file_path)
    # print(pd.read_csv(file_path).head())