import pandas as pd
from sklearn.model_selection import ParameterGrid
import numpy as np
from enum import Enum


class DataEnum(Enum):
    FISHING = 'FISHING'
    HURRICANES = 'HURRICANES'
    GEOLIFE = 'GEOLIFE'


class AlgoEnum(Enum):
    CBSMoT = 'CB-SMoT'
    DBSMoT = 'DB-SMoT'


def get_data(d, algorithm):
    ret = None
    if d == DataEnum.FISHING:
        data = get_fv()
        label = 'label'
    elif d == DataEnum.HURRICANES:
        data = get_hurr()
        label = 'label'
    elif d == DataEnum.GEOLIFE:
        data = get_geolife()
        label = 'transportation_mode'
    if algorithm == AlgoEnum.CBSMoT:
        parms = cbsmot_parms(d)
    elif algorithm == AlgoEnum.DBSMoT:
        parms = dbsmot_parms(d)
    ret = {'data': data,
           'parameter_grid': parms,
           'file_name': str(algorithm.value) + "_" + str(d.value) + "_results.csv",
           'label': label}
    return ret


def get_fv(path='cd Vessels/'):
    dfs = []
    dfs.append(split_df(pd.read_csv(path + 'fv_d1.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d2.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d3.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d4.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d5.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d6.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d7.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d8.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d9.txt', sep=';')))
    dfs.append(split_df(pd.read_csv(path + 'fv_d10.txt', sep=';')))
    return dfs


def get_hurr():
    dfs = []
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d1.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d2.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d3.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d4.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d5.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d6.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d7.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d8.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d9.txt', sep=';')))
    dfs.append(split_df(pd.read_csv('databases/Hurricanes/h_d10.txt', sep=';')))
    return dfs


def get_geolife():
    dfs = []
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_1.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_2.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_3.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_4.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_5.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_6.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_7.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_8.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_9.csv', sep=',')])
    dfs.append([pd.read_csv('databases/geolife/geolife_w_features_10.csv', sep=',')])
    return dfs


def cbsmot_parms(d):
    ret = None
    if d == DataEnum.HURRICANES:
        ret = ParameterGrid({'area': list(np.arange(0.1, 1, 0.025)),
                             'min_time': np.array(range(0, 24, 6)) * 3600,
                             'time_tolerance': [0],
                             'merge_tolerance': [0]})
    elif d == DataEnum.FISHING:
        ret = ParameterGrid({'area': list(np.arange(0.1, 1, 0.05)),
                             'min_time': np.array(range(2, 13, 2)) * 3600,
                             'time_tolerance': [0],
                             'merge_tolerance': [0]})
    elif d == DataEnum.GEOLIFE:
        ret = ParameterGrid({'area': np.arange(0.1, 0.9, 0.05),
                             'min_time': np.arange(0.1, 3, 0.1) * 3600,
                             'time_tolerance': [0],
                             'merge_tolerance': [0]})
    return ret


def dbsmot_parms(d):
    ret = None
    if d == DataEnum.HURRICANES:
        ret = ParameterGrid({})
    elif d == DataEnum.FISHING:
        ret = ParameterGrid({})
    elif d == DataEnum.GEOLIFE:
        ret = ParameterGrid({})
    return ret


def split_df(df, label='tid'):
    real_dfs = []
    df.set_index(keys=['tid'], drop=False, inplace=True)
    tids = df['tid'].unique().tolist()
    for tid in tids:
        real_dfs.append(df.loc[df.tid == tid])
    return real_dfs


def read_fold(file_name, new_columns_names, verbose=False):
    if verbose:
        print("Mandatory fields:[lat,lon,time,label,sid]")
        print("Remove duplicates on time")
        print("Index on time")
    df1 = pd.read_csv(file_name, sep=';', parse_dates=['time'])

    df1.rename(columns=new_columns_names, inplace=True)
    tsid_ = df1.sid * 10000 + df1.tid
    df1 = df1.assign(TSid=tsid_)
    #df1 = df1.loc[:, ['lat', 'lon', 'time', 'label', 'TSid']]

    df1.drop_duplicates(subset=['TSid', 'time'], keep=False, inplace=True)

    df1.sort_values(by=['TSid', 'time'], inplace=True)

    assert np.isin(['lat', 'lon', 'time', 'label', 'TSid'], list(df1.columns)).all(),\
        'We need all fields: [lat,lon,time,label,TSid]'
    df1 = df1.set_index(['time'])

    return df1.copy()

def read_fold3(file_name, new_columns_names, verbose=False):
    if verbose:
        print("Mandatory fields:[lat,lon,time,label,sid]")
        print("Remove duplicates on time")
        print("Index on time")
    df1 = pd.read_csv(file_name, sep=';', parse_dates=['time'])


    df1.rename(columns=new_columns_names, inplace=True)
    tsid_ = df1.sid * 10000 + df1.tid
    df1 = df1.assign(TSid=tsid_)
    df1 = df1.loc[:, ['lat', 'lon', 'time', 'label', 'TSid']]

    df1.drop_duplicates(subset=['TSid', 'time'], keep=False, inplace=True)

    df1.sort_values(by=['TSid', 'time'], inplace=True)

    assert np.isin(['lat', 'lon', 'time', 'label', 'TSid'], list(df1.columns)).all(),\
        'We need all fields: [lat,lon,time,label,TSid]'
    df1 = df1.set_index(['time'])

    return df1.copy()

def load_data_hurricane_data(path='/projects/trajlib_v2/databases/Hurricanes'):
    print(path)
    new_columns_names = {'latitude': 'lat', 'longitude': 'lon'}
    df_fold1 = read_fold(path + '/h_d1.txt', new_columns_names)
    df_fold2 = read_fold(path + '/h_d2.txt', new_columns_names)
    df_fold3 = read_fold(path + '/h_d3.txt', new_columns_names)
    df_fold4 = read_fold(path + '/h_d4.txt', new_columns_names)
    df_fold5 = read_fold(path + '/h_d5.txt', new_columns_names)
    df_fold6 = read_fold(path + '/h_d6.txt', new_columns_names)
    df_fold7 = read_fold(path + '/h_d7.txt', new_columns_names)
    df_fold8 = read_fold(path + '/h_d8.txt', new_columns_names)
    df_fold9 = read_fold(path + '/h_d9.txt', new_columns_names)
    df_fold10 = read_fold(path + '/h_d10.txt', new_columns_names)
    return [df_fold1, df_fold2, df_fold3, df_fold4, df_fold5, df_fold6, df_fold7, df_fold8, df_fold9, df_fold10]


def read_fold_mts(file_name,  verbose=False):
    if verbose:
        print("Mandatory fields:[lat,lon,time,label,sid]")
        print("Remove duplicates on time")
        print("Index on time")
    df1 = pd.read_csv(file_name, parse_dates=['time'])
    #print(df1.shape)

    df1.drop_duplicates(subset=['tsid', 'time'], keep='first', inplace=True)
    #print(df1.shape)
    df1.sort_values(by=['tsid', 'time'], inplace=True)

    assert np.isin(['lat', 'lon', 'time', 'label', 'tsid'], list(df1.columns)).all(),\
        'We need all fields: [lat,lon,time,label,tsid]'
    df1.rename(columns={'tsid':'TSid'}, inplace=True)
    df1 = df1.set_index(['time'])
    df1=df1.sort_index()
    return df1.copy()

def load_data_fishing_data_fishing2(path='/projects/trajlib_v2/databases/fishing2'):
    print(path)
    df_fold1 = read_fold_mts(path + '/Fishing_1.csv' )
    df_fold2 = read_fold_mts(path + '/Fishing_2.csv' )
    df_fold3 = read_fold_mts(path + '/Fishing_3.csv' )
    df_fold4 = read_fold_mts(path + '/Fishing_4.csv' )
    df_fold5 = read_fold_mts(path + '/Fishing_5.csv' )
    df_fold6 = read_fold_mts(path + '/Fishing_6.csv' )
    df_fold7 = read_fold_mts(path + '/Fishing_7.csv' )
    df_fold8 = read_fold_mts(path + '/Fishing_8.csv' )
    df_fold9 = read_fold_mts(path + '/Fishing_9.csv')
    df_fold10 = read_fold_mts(path + '/Fishing_10.csv')
    df_fold11 = read_fold_mts(path + '/Fishing_11.csv')
    df_fold12 = read_fold_mts(path + '/Fishing_12.csv')
    df_fold13 = read_fold_mts(path + '/Fishing_13.csv')
    df_fold14 = read_fold_mts(path + '/Fishing_14.csv')
    df_fold15 = read_fold_mts(path + '/Fishing_15.csv')
    df_fold16 = read_fold_mts(path + '/Fishing_16.csv')
    return [df_fold1, df_fold2, df_fold3, df_fold4, df_fold5,
            df_fold6, df_fold7, df_fold8, df_fold9, df_fold10
            , df_fold11, df_fold12, df_fold13, df_fold14
            , df_fold15, df_fold16]

def load_data_fishing_data_mts(path='/projects/trajlib_v2/databases/fishing'):
    print(path)
    df_fold1 = read_fold_mts(path + '/fv_mts_d1.csv' )
    df_fold2 = read_fold_mts(path + '/fv_mts_d2.csv' )
    df_fold3 = read_fold_mts(path + '/fv_mts_d3.csv' )
    df_fold4 = read_fold_mts(path + '/fv_mts_d4.csv' )
    df_fold5 = read_fold_mts(path + '/fv_mts_d5.csv' )
    df_fold6 = read_fold_mts(path + '/fv_mts_d6.csv' )
    df_fold7 = read_fold_mts(path + '/fv_mts_d7.csv' )
    df_fold8 = read_fold_mts(path + '/fv_mts_d8.csv' )
    df_fold9 = read_fold_mts(path + '/fv_mts_d9.csv')
    df_fold10 = read_fold_mts(path + '/fv_mts_d10.csv')
    return [df_fold1, df_fold2, df_fold3, df_fold4, df_fold5, df_fold6, df_fold7, df_fold8, df_fold9, df_fold10]


def load_data_fishing_data_recons(path='/projects/trajlib_v2/databases/fishing'):
    print(path)
    df_fold1 = read_fold_mts(path + '/fv_d_recons1.csv' )
    df_fold2 = read_fold_mts(path + '/fv_d_recons2.csv' )
    df_fold3 = read_fold_mts(path + '/fv_d_recons3.csv' )
    df_fold4 = read_fold_mts(path + '/fv_d_recons4.csv' )
    df_fold5 = read_fold_mts(path + '/fv_d_recons5.csv' )
    df_fold6 = read_fold_mts(path + '/fv_d_recons6.csv' )
    df_fold7 = read_fold_mts(path + '/fv_d_recons7.csv' )
    df_fold8 = read_fold_mts(path + '/fv_d_recons8.csv' )
    df_fold9 = read_fold_mts(path + '/fv_d_recons9.csv')
    df_fold10 = read_fold_mts(path + '/fv_d_recons10.csv')
    return [df_fold1, df_fold2, df_fold3, df_fold4, df_fold5, df_fold6, df_fold7, df_fold8, df_fold9, df_fold10]

def load_data_fishing_data(path='/projects/trajlib_v2/databases/fishing'):
    print(path)
    new_columns_names = {'latitude': 'lat', 'longitude': 'lon'}
    df_fold1 = read_fold(path + '/fv_d1.txt', new_columns_names)
    df_fold2 = read_fold(path + '/fv_d2.txt', new_columns_names)
    df_fold3 = read_fold(path + '/fv_d3.txt', new_columns_names)
    df_fold4 = read_fold(path + '/fv_d4.txt', new_columns_names)
    df_fold5 = read_fold(path + '/fv_d5.txt', new_columns_names)
    df_fold6 = read_fold(path + '/fv_d6.txt', new_columns_names)
    df_fold7 = read_fold(path + '/fv_d7.txt', new_columns_names)
    df_fold8 = read_fold(path + '/fv_d8.txt', new_columns_names)
    df_fold9 = read_fold(path + '/fv_d9.txt', new_columns_names)
    df_fold10 = read_fold(path + '/fv_d10.txt', new_columns_names)
    return [df_fold1, df_fold2, df_fold3, df_fold4, df_fold5, df_fold6, df_fold7, df_fold8, df_fold9, df_fold10]

def load_data_AIS_data(path='~/Trajlib2/Trajlib2/databases/ais/'):
    print(path)
    new_columns_names = {'location.coordinates.0': 'lat', 'location.coordinates.1': 'lon', 'mmsi': 'label',
                         'event_time':'time','sog':'lid'}
    df_fold1 = read_fold4(path + '/v_1.csv', new_columns_names)
    df_fold2 = read_fold4(path + '/v_2.csv', new_columns_names)
    df_fold3 = read_fold4(path + '/v_3.csv', new_columns_names)
    df_fold4 = read_fold4(path + '/v_4.csv', new_columns_names)
    df_fold5 = read_fold4(path + '/v_5.csv', new_columns_names)
    df_fold6 = read_fold4(path + '/v_6.csv', new_columns_names)
    df_fold7 = read_fold4(path + '/v_7.csv', new_columns_names)
    df_fold8 = read_fold4(path + '/v_8.csv', new_columns_names)
    df_fold9 = read_fold4(path + '/v_9.csv', new_columns_names)
    df_fold10 = read_fold4(path + '/v_10.csv', new_columns_names)
    return [df_fold1, df_fold2, df_fold3, df_fold4, df_fold5, df_fold6, df_fold7, df_fold8, df_fold9, df_fold10]

def load_data_geolife_data(path='~/Trajlib2/Trajlib2/databases/geolife/'):
    print(path)
    new_columns_names = {'latitude': 'lat', 'longitude': 'lon', 'transportation_mode': 'label'}
    df_fold1 = read_fold2(path + '/geolife_w_features_1.csv', new_columns_names)
    df_fold2 = read_fold2(path + '/geolife_w_features_2.csv', new_columns_names)
    df_fold3 = read_fold2(path + '/geolife_w_features_3.csv', new_columns_names)
    df_fold4 = read_fold2(path + '/geolife_w_features_4.csv', new_columns_names)
    df_fold5 = read_fold2(path + '/geolife_w_features_5.csv', new_columns_names)
    df_fold6 = read_fold2(path + '/geolife_w_features_6.csv', new_columns_names)
    df_fold7 = read_fold2(path + '/geolife_w_features_7.csv', new_columns_names)
    df_fold8 = read_fold2(path + '/geolife_w_features_8.csv', new_columns_names)
    df_fold9 = read_fold2(path + '/geolife_w_features_9.csv', new_columns_names)
    df_fold10 = read_fold2(path + '/geolife_w_features_10.csv', new_columns_names)
    return [df_fold1, df_fold2, df_fold3, df_fold4, df_fold5, df_fold6, df_fold7, df_fold8, df_fold9, df_fold10]

"""
def read_fold2(file_name, new_columns_names, by_=None, sep_=','):
    if by_ is None:
        by_ = ['tsid', 'time']
    df1 = pd.read_csv(file_name, sep=sep_, parse_dates=['time'])
    df1 = df1.sort_index()
    # df1=df1.sort_values(by=['time'], inplace=True)
    # df1.set_index('time', inplace=True)
    df1.rename(columns=new_columns_names, inplace=True)
    # df1.columns.values[df1.columns == 'latitude'] = 'lat'
    # df1.columns.values[df1.columns == 'longitude'] = 'lon'
    # tsid_ = df1.sid * 10000 + df1.tid
    tsid_ = df1.lid
    df_fold1 = df1.assign(tsid=tsid_)
    df_fold1.sort_values(by=by_, inplace=True)
    return df_fold1.copy()
"""
def tz_to_naive(datetime_index):
    """Converts a tz-aware DatetimeIndex into a tz-naive DatetimeIndex,
    effectively baking the timezone into the internal representation.

    Parameters
    ----------
    datetime_index : pandas.DatetimeIndex, tz-aware

    Returns
    -------
    pandas.DatetimeIndex, tz-naive
    """
    # Calculate timezone offset relative to UTC
    timestamp = datetime_index[0]
    from datetime import datetime
    #print(type(datetime_index))
    if type(datetime_index.values[0]) is datetime:
        datetime_index=pd.Series(np.array([np.datetime64(x) for x in datetime_index.values]))
    if type(timestamp) is datetime:
        timestamp=pd.Timestamp(timestamp)
        #print(type(timestamp), type(datetime_index.values),type(datetime_index.values[0]))
    else:
        #print(type(timestamp),type(datetime_index.values),type(datetime_index.values[0]))
        pass
    tz_offset = (timestamp.replace(tzinfo=None) -
                 timestamp.tz_convert('UTC').replace(tzinfo=None))

    tz_offset_td64 = np.timedelta64(tz_offset)



    #print(tz_offset_td64)
    # Now convert to naive DatetimeIndex
    #return pd.DatetimeIndex(datetime_index.values + tz_offset_td64
    return pd.DatetimeIndex(datetime_index.values )

def read_fold2(file_name, new_columns_names, sep_=',', verbose=False):
    if verbose:
        print("Mandatory fields:[lat,lon,time,label,sid]")
        print("Remove duplicates on time")
        print("Index on time")
    col_time='time'
    #a=dict((v, k) for k, v in new_columns_names.items())[col_time]
    #if col_time in new_columns_names.values():
    #    col_time=a
    df1 = pd.read_csv(file_name, sep=sep_, parse_dates=[col_time])
    df1.rename(columns=new_columns_names, inplace=True)
    if df1['time'][0].tzinfo!=None:
        df1['time'] = tz_to_naive(df1['time'])

    tsid_ = df1.lid
    df1 = df1.assign(TSid=tsid_)
    df1 = df1.loc[:, ['lat', 'lon', 'time', 'label', 'TSid']]

    ff=df1.shape[0]
    df1.drop_duplicates(subset=['time'], keep='first', inplace=True)
    if (df1.shape[0]!=ff):
        print('duplicates removed:',ff-df1.shape[0],' from ',file_name)
    df1.sort_values(by=[ 'time'], inplace=True)

    assert np.isin(['lat', 'lon', 'time', 'label', 'TSid'], list(df1.columns)).all(),\
        'We need all fields: [lat,lon,time,label,TSid]'
    df1 = df1.set_index(['time'])
    df1 = df1.loc[~df1.index.duplicated(keep='first')]
    df1.sort_index(inplace=True)

    return df1.copy()
def read_fold4(file_name, new_columns_names, sep_=',', verbose=False):
    if verbose:
        print("Mandatory fields:[lat,lon,time,label,sid]")
        print("Remove duplicates on time")
        print("Index on time")
    col_time='time'
    a=dict((v, k) for k, v in new_columns_names.items())['time']
    if col_time in new_columns_names.values():
        col_time=a
    df1 = pd.read_csv(file_name, sep=sep_, parse_dates=[col_time])
    df1.rename(columns=new_columns_names, inplace=True)
    if df1['time'][0].tzinfo!=None:
        df1['time'] = tz_to_naive(df1['time'])

    tsid_ = df1.lid
    df1 = df1.assign(TSid=tsid_)
    df1 = df1.loc[:, ['lat', 'lon', 'time', 'label', 'TSid']]

    ff=df1.shape[0]
    df1.drop_duplicates(subset=['time'], keep='first', inplace=True)
    if (df1.shape[0]!=ff):
        print('duplicates removed:',ff-df1.shape[0],' from ',file_name)
    df1.sort_values(by=[ 'time'], inplace=True)

    assert np.isin(['lat', 'lon', 'time', 'label', 'TSid'], list(df1.columns)).all(),\
        'We need all fields: [lat,lon,time,label,TSid]'
    df1 = df1.set_index(['time'])
    df1 = df1.loc[~df1.index.duplicated(keep='first')]
    df1.sort_index(inplace=True)

    return df1.copy()