from itertools import Predicate
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer , train_test_split
lll = {'LABEL', 'ICO_CUR_MON_ACM_TRX_TM', 'NB_RCT_3_MON_LGN_TMS_AGV', 'ICO_CUR_MON_ACM_TRX_AMT', 'COUNTER_CUR_YEAR_CNT_AMT', 'PUB_TO_PRV_TRX_AMT_CUR_YEAR', 'MON_12_EXT_SAM_TRSF_IN_AMT', 'MON_12_EXT_SAM_TRSF_OUT_AMT', 'MON_12_EXT_SAM_NM_TRSF_OUT_CNT', 'MON_12_EXT_SAM_AMT', 'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT', 'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT', 'CUR_YEAR_MON_AGV_TRX_CNT', 'MON_12_AGV_TRX_CNT', 'MON_12_ACM_ENTR_ACT_CNT', 'MON_12_AGV_ENTR_ACT_CNT', 'MON_12_ACM_LVE_ACT_CNT', 'MON_12_AGV_LVE_ACT_CNT', 'CUR_YEAR_COUNTER_ENCASH_CNT', 'LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL', 'LAST_12_MON_COR_DPS_DAY_AVG_BAL', 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL', 'CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL', 'CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR', 'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV', 'LAST_12_MON_MON_AVG_TRX_AMT_NAV', 'COR_KEY_PROD_HLD_NBR', 'CUR_YEAR_MID_BUS_INC', 'AI_STAR_SCO', 'EMP_NBR', 'REG_CPT', 'SHH_BCK', 'HLD_DMS_CCY_ACT_NBR', 'REG_DT', 'OPN_TM', 'HLD_FGN_CCY_ACT_NBR'}
idx = pd.Index(lll)
def wash_raw(df,lll=lll):
    idx = pd.Index(lll)
    df.replace('?',np.NaN,inplace=True)
    df['MON_12_CUST_CNT_PTY_ID'].fillna(0,inplace=True)
    df[df['MON_12_CUST_CNT_PTY_ID']=='Y']=1
    mask = df[:].dtypes != object
    df = df.loc[:,mask]
    return df.loc[:,df.columns.isin(idx)]

testData = pd.read_csv('/Users/fyq/PycharmProjects/CMB/数据赛道/test_A榜.csv',encoding='utf-8')
trainData = pd.read_csv('/Users/fyq/PycharmProjects/CMB/数据赛道/fillna_knn/fillnaAVG.csv',index_col=0)
trainLabel = trainData.loc[:,'LABEL']
testData = wash_raw(testData)
trainData = trainData.loc[:,trainData.columns.isin(testData.columns)]

# fill nan

imputer = KNNImputer(n_neighbors=10, weights='uniform')
fillna_train = imputer.fit_transform(trainData)
testData = imputer.fit_transform(testData)
f = open('fillna.csv','a')

#  全部knn
def all_KNN(x = fillna_train,y=trainLabel):
    neigh = KNeighborsRegressor(n_neighbors=10)
    neigh.fit(x,y)
    return neigh.predict(testData)
    
preidct_value = all_KNN(fillna_train,y=trainLabel)
pd.DataFrame(preidct_value).to_csv('knn-result.csv')

################################################
#回测KNN

def look_back_knn():
    #split
