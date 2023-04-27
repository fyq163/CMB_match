import datetime
import itertools
import random
from datetime import time

import pandas as pd
import numpy as np


def test_ts_metal(w=None):
    import tensorflow as tf

    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=w,
        input_shape=(32, 32, 3),
        classes=100,)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)


def dynamic_string_col(df, param_string_col=None):
    """
     Get list of string columns that are used to build dynamic data. This is a helper function for : func : ` ~pyspark. sql. query. get_or_create_sql `

     @param df - pandas. DataFrame Data frame to search
     @param param_string_col - list of strings to include in result

     @return list of string columns to be used in SQL query
    """

    # param_string_col is None if not set.
    if param_string_col is None:
        param_string_col = ['MON_12_CUST_CNT_PTY_ID',
                            'AI_STAR_SCO',
                            'WTHR_OPN_ONL_ICO',
                            'SHH_BCK',
                            'LGP_HLD_CARD_LVL',
                            'NB_CTC_HLD_IDV_AIO_CARD_SITU']
    _string_col = list(set(df.columns) & set(param_string_col))
    return _string_col


def drop_too_few_variable(dataframe: pd.DataFrame, threshold=0.75, keep_stringcol=True):
    mask_drop = (dataframe.count() / 40000) > threshold
    if keep_stringcol:
        _df_numerical = dataframe.loc[:, mask_drop]
        return _df_numerical
    else:
        dataframe = dataframe.loc[:, mask_drop]


def mode_nan_string(mode_df):
    the_mode = mode_df.mode()
    for col in mode_df:  # the mode of that column
        mode_df.loc[:, col].fillna(the_mode.loc[0, col], inplace=True)
    return mode_df


def random_forest_nan(df):
    pass


def step1_data_processing(df: pd.DataFrame, _string_col):
    """

    minus 2 for numerical columns and 'SHH_BCK' column, then separate label dataframe

    :param _string_col: none numerical columns

    :param df : dataframe to be processed

    :returns _string_col: dataframe after step 1 processing , a pure label

    """
    # slip 2 numerical and string columns
    _string_col = list(set(df.columns) & set(_string_col))
    # reconfirm the columns
    _df_numerical = df.drop(_string_col, axis=1).iloc[:, 2:].astype(float)
    _df_numerical = _df_numerical - 2
    df_str = df.loc[:, _string_col]
    df_str.loc[:, 'SHH_BCK'] = df.loc[:, 'SHH_BCK'] - 2
    return pd.concat(objs=[df.iloc[:, :2], _df_numerical, df_str], axis=1), _string_col


def get_numerical_df(df: pd.DataFrame, _drop_col=None):
    """
    Takes df and then return all numerical data in Dataframe



    Returns
    -------
    pd.DataFrame

    :return df_numerical: dataframe with all numerical data
    """

    if _drop_col is None:
        _drop_col = ['CUST_UID', 'LABEL']
    _string_col = dynamic_string_col(df)
    _df_numerical = df.drop(_string_col, axis=1)
    if any(_df_numerical.columns.isin(['CUST_UID', 'LABEL'])):
        return _df_numerical.drop(
            _drop_col, axis=1).astype(float)
    else:
        return _df_numerical


def create_dummies(df_raw, binary_col=['MON_12_CUST_CNT_PTY_ID']) -> pd.DataFrame:
    """
    :df_raw : df with numerical cols and str cols
    """
    binary_col = list(binary_col)
    string_col = dynamic_string_col(df_raw)
    opt_col = string_col.copy()
    df_str = pd.DataFrame()
    for i in binary_col:
        if i in string_col:
            # This a Binary Y/N, the training sample has Y only, do not fill with the mode
            opt_col.remove(i)
            df_str.loc[:, i] = df_raw.loc[:, i]
        # Direct copy
    # fill the missing value with mode
    df_str.loc[:, opt_col] = mode_nan_string(df_raw.loc[:, opt_col])
    # drop opt_col, later rejoin in
    df_raw.drop(string_col, axis=1, inplace=True)
    # One-hot encoding
    dummies = pd.get_dummies(df_str, dummy_na=True).astype(int)
    return pd.concat([df_raw, dummies], axis=1)


def drop_high_corr_columns(df_raw: pd.DataFrame, to_calc_corr: pd.DataFrame, threshold: object = 0.999) -> list:
    """
    Get corr matrix from to_calc_corr, which is a dataframe without any string value,
    Then filter by number of NaN in df_raw, save the minium then compare between paris
    Finally return a list of column name to be dropped

    :param to_calc_corr: This is the dataframe used to calculate the correlation, it should a Dataframe without any string value
    :param df_raw: to statistic the number of NaN values
    :param threshold: threshold to distinguish what is left what to be droed
    :rtype: list
    :return: a list of column name to be dropped
    """
    corr_matrix = to_calc_corr.corr()
    rows, cols = np.where(corr_matrix > 0.8)
    get_rid_2 = list()
    for i, row in enumerate(rows):
        position = corr_matrix.iloc[row, cols[i]]
        # rows[i],cols[i] is the position of the element in the matrix
        # posi is the correlation value of rows[i] and cols[i]
        if position < threshold:  # if the correlation value is less than threshold
            col_set = {corr_matrix.index[rows[i]],
                       corr_matrix.columns[cols[i]]}
            # this is the actual column name
            if col_set not in get_rid_2:  # because the set disregard the order,
                # so we need to check if the set is already in the list,
                # the pair with the same two elements
                # but different order will be regarded as same set,
                get_rid_2.append(col_set)  # ----> a list of set
    drop_col_2 = list(set([item for s in get_rid_2 for item in list(s)]))
    # --> a list of column name, all high correlation columns
    rank_l = []
    for i, s in enumerate(get_rid_2):
        for j, col_j in enumerate(s):
            rank_l.append([col_j,
                           # full column name
                           col_j[0:3],
                           # abbreviation
                           i,
                           # pair number
                           df_raw.loc[:, col_j].isna().sum()]
                          # number of missing values
                          )
    rank_df = pd.DataFrame(np.array(rank_l),
                           columns=['col name', 'abbreviation', 'pair', 'number of NaN'])
    rank_df['number of NaN'].astype('int64')
    rank_df['pair'].astype('string')

    # DON'T DELETE, USE TO QUERY
    # print(rank_df.loc[rank_df.loc[:,'pair']=='1'])
    abb = rank_df.sort_values(['number of NaN', 'pair'],
                              ascending=[True, False]) \
        .groupby(
        'abbreviation') \
        .min() \
        .groupby('pair').min()

    #                             try :
    #                             .head(1 )
    save_col = abb['col name'].tolist()
    #                      high corr columns               if not in result cols
    return [col for col in to_calc_corr[drop_col_2].columns if col not in save_col]

    # first take list get nan number
    # second  get abbreviation, take set get pair
    # save the least nan in each abbreviation
    # save the least nan in each pair
    # get the cols we want to save
    # drop the rest
    # df.nlargest()Z


def random_check_order(new: pd.DataFrame, old: pd.DataFrame):
    """
    check match,
    random select from range first few row,col
    see if the value math

    :param new: new dataframe
    :param old: benchmark dataframe, the old one
    :return: a prompt
    """
    res_list = []
    for i in range(500):
        row_i = random.randint(20, 300)
        col_i = random.randint(2, 5)
        # random coordinate to get a row_n, col_name
        row_name = new.index[row_i]
        col_name = new.columns[col_i]
        # col, row name is the same for new and old dataframe
        n_value = new.loc[row_name, col_name]
        o_value = old.loc[row_name, col_name]
        # any() only check if there is true in the array, so reverse the boolean
        res_list.append(n_value != o_value)
    if not any(res_list):  # any abnormal?
        # --->TF array
        # T/F array.any()
        return 'great, no difference'


def IQR_filtering(df, U=0.88, L=0.22):
    """
    Optional
    ----------
    This function filters outliers using IQR method.

    Parameters
    ----------
    df : pd.Dataframe,
        This dataframe should be numerial only, if there is a string column, it will crash

    U : default is 0.88
        percentile of the upper bound, default is 0.88

    L : default is 0.22
        percentile of the lower bound, default is 0.22

    """

    for col in df.columns:
        Q1 = df.loc[:, col].quantile(L)
        Q3 = df.loc[:, col].quantile(U)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        con_cahnge = (df[col] < lower) | (df[col] > upper)

        def change(x):
            if x < lower:
                return lower
            elif x > upper:
                return upper
            else:
                return x

        df.loc[con_cahnge, col] = df.loc[con_cahnge, col].apply(change)
    return df


if __name__ == '__main__':
    import os
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.split(current_directory)[0] # Repeat as needed
    

    path_fill = 'data/2022/filled_trian.csv'
    path_fill = os.path.join(parent_directory,path_fill)
    path_train = os.path.join(parent_directory,'data/2022/train.csv')
    df_train = pd.read_csv(path_train, index_col=0).head(500)
    df_train_filled = pd.read_csv(path_fill, index_col=0).head(500)
    # upper lower limits should there be at the first place

    # drop corr FIRST of fill mode&OneHot FIRST?
    y_sample_500 = df_train_filled['LABEL']
    x_sample_500 = create_dummies(df_train_filled).drop(['CUST_UID', 'LABEL'], axis=1)
    drop_col: list = drop_high_corr_columns(df_train, x_sample_500)
    x_sample_500.drop(drop_col, axis=1, inplace=True)

    test = df_train_filled[['COUNTER_CUR_YEAR_CNT_AMT',
                            'CUR_YEAR_COUNTER_ENCASH_CNT',
                            'HLD_FGN_CCY_ACT_NBR']]
    # print(get_numerical_df(test))
    test_ts_metal()