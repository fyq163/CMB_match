import pandas as pd


def dynamic_string_col(df, param_string_col=None):
    if param_string_col is None:
        param_string_col = ['MON_12_CUST_CNT_PTY_ID',
                            'AI_STAR_SCO',
                            'WTHR_OPN_ONL_ICO',
                            'SHH_BCK',
                            'LGP_HLD_CARD_LVL',
                            'NB_CTC_HLD_IDV_AIO_CARD_SITU']
    _string_col = list(set(df.columns) & set(param_string_col))
    return _string_col


def mode_nan_string(mode_df):
    the_mode = mode_df.mode()
    for col in mode_df:
        mode_df.loc[:, col].fillna(the_mode.loc[0, col], inplace=True)
    return mode_df


def random_forest_nan(df):
    return df


def drop_too_few_variable(dataframe: pd.DataFrame, threshold=0.75, keep_stringcol=True):
    mask_drop = (dataframe.count() / 40000) > threshold
    if keep_stringcol:
        _df_numerical = dataframe.loc[:, mask_drop]
    else:
        dataframe = dataframe.loc[:, mask_drop]
    return _df_numerical


def step1_data_processing(df: pd.DataFrame, _string_col):
    """
    minus 2 for numerical columns and 'SHH_BCK' column, then separate label dataframe
    :param _string_col: none numerical columns
    :returns dataframe after step 1 processing , a pure label
    """
    # slip 2 numerical and string columns
    _string_col = list(set(df.columns) & set(_string_col))
    # reconfirm the columns
    _df_numerical = df.drop(_string_col, axis=1).iloc[:, 2:].astype(float)
    _df_numerical = _df_numerical - 2
    df_str = df.loc[:, _string_col]
    df_str.loc[:, 'SHH_BCK'] = df.loc[:, 'SHH_BCK'] - 2
    return pd.concat(objs=[df.iloc[:, :2], _df_numerical, df_str], axis=1), _string_col


def get_numerical_df(df: pd.DataFrame, drop_col=None):
    """
    Takes df and then return all numerical data in Dataframe
    :return df_numerical
    """
    if drop_col is None:
        drop_col = ['CUST_UID', 'LABEL']
    else:
        drop_col = drop_col
    _string_col = dynamic_string_col(df)
    _df_numerical = df.drop(_string_col, axis=1).drop(drop_col, axis=1).astype(float)
    return _df_numerical


def high_corr_col(df: pd.DataFrame, threshold=0.8) -> list:
    """
    :return a list contain high correlation columns
    :param df:
    :param threshold: threshold to distinguish what is left what to be droed
    :rtype: list
    """
    # get all numerical
    df_numerical = get_numerical_df(df)
    # get correlation matrix
    corr_matrix = df_numerical.corr()
    corr_cols = []
    # high corr columns
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j] > threshold):
                if corr_matrix.columns[j] not in corr_cols:
                    corr_cols.append(corr_matrix.columns[j])
    return corr_cols


def save_n_drop(corr_cols, df: pd.DataFrame):
    """
    :param corr_cols: The candidate cols you want to drop
    :param df: Raw dataframe that we can be counting NaN from
    """
    # from big df get the number of NaN of each column
    rank_df = pd.DataFrame(df.loc[:, corr_cols].isna().sum())
    # for each abbreviation we keep 1, we assume it is the same thing
    rank_df.loc[:, 'abbreviation'] = rank_df.index.map(lambda x: x[0:3])
    # for each group of abbreviation, we keep only the fullest data set
    rank_res_col = rank_df.sort_values([0], ascending=True).groupby('abbreviation').head(1).index
    return [col for col in df[corr_cols].columns if col not in rank_res_col]


# if 'LABEL' in df.columns:
#         df_label: pd.DataFrame = df_train.LABEL
#     else:
#         raise ValueError('There is no label!')
if __name__ == '__main__':
    df_train = pd.read_csv('filled_trian.csv', index_col=0).head(100)
    string_col = dynamic_string_col(df_train)
