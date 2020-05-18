import pandas as pd
import numpy as np

def convert_mts(filename="~/Trajlib2/Trajlib2/databases/fishing/fv_d3.txt",sep=';'):
    df=pd.read_csv(filename,sep=sep)
    print( df.shape)
    df=df.assign(dt=pd.to_datetime(df.time))
    df=df.set_index(['tid','dt'])
    df=df.sort_index()
    df=df.reset_index()

    min_date=pd.to_datetime(np.min(df.time))
    max_date=pd.to_datetime(df.time.max())
    #print(min_date,max_date)
    end_traj=max_date+pd.Timedelta(1000000000*60*60)
    res=pd.DataFrame()
    for k,v in df.groupby(['tid']):
        if end_traj>=pd.to_datetime(np.min(v.time)):
            #print("v ok")
            #print("o", np.min(v.time), np.max(v.time))
            end_traj = pd.to_datetime(v.time.max()) + pd.Timedelta(1000000000 * 60 * 60)
            res2=v.assign(new_time=pd.to_datetime(v.time))
            res = res.append(res2)
        else:
            new_date=end_traj-pd.to_datetime(np.min(v.time))
            new_time=pd.to_datetime(v.time)+new_date
            v=v.assign(new_time=new_time)
            res = res.append(v)
            #print("o", np.min(v.time), np.max(v.time))
            #print(k,np.min(v.new_time),np.max(v.new_time))
            end_traj=v.new_time.max()+pd.Timedelta(1000000000*60*60)

    res=res.drop(columns=['time'])
    tsid_ = res.sid * 10000 + res.tid
    res=res.assign(tsid=tsid_)
    res=res.loc[:,['tsid', 'latitude', 'longitude','label', 'new_time']]
    res.columns=['tsid', 'lat', 'lon','label', 'time']
    print( res.shape)
    return res
for i in range(10):
    df=convert_mts("~/Trajlib2/Trajlib2/databases/fishing/fv_d"+str(i+1)+".txt")
    df.to_csv('~/Trajlib2/Trajlib2/databases/fishing/fv_mts_d'+str(i+1)+'.csv')

