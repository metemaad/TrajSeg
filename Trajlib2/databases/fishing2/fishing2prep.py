import pandas as pd
import numpy as np
from Trajlib2.TrajectorySegmentation import TrajectorySegmentation
df=pd.read_csv('fishing.csv')
dic={'rid':'rid', 'COARSE_FIS':'label', 'latitude':'lat', 'longitude':'lon', 'MMSI':'MMSI'
        , 'SOG':'SOG',
       'collected_time':'time1'}
df.rename(columns=dic, inplace=True)
print(df.columns)
s=TrajectorySegmentation(df)
sid=s.segment_by_label('label')
df=df.assign(tsid=np.array(sid).astype(int))
tima=pd.to_datetime(df.time1)
df=df.assign(time=np.array(tima))
print(df.head())
print(list(set(df.MMSI)))
i=1
for k,v in df.groupby(['MMSI']):
    v=v.loc[:,['lat', 'lon', 'tsid', 'label', 'time']]
    v=v.sort_values(['time'])
    v.to_csv("Fishing_"+str(i)+".csv")
    i=i+1
   #, lat, lon, tsid, label, time