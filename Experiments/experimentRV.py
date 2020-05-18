from Trajlib2.databases import load_datasets
from Trajlib2.TrajectorySegmentation import TrajectorySegmentation
from Trajlib2.SegmentationAlgorithms.SWS.SegmentBySWS import SegmentationBySWS
import Trajlib2.SegmentationEvaluation as segmentation_evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import psutil
import random
from Trajlib2.SegmentationAlgorithms.SWS.sws import SWS
from Experiments.Fishing.plot import plot_segments
from Trajlib2.core.traj_reconstruct import plot_df
ds = load_datasets.load_data_fishing_data_mts('~/Trajlib2/Trajlib2/databases/fishing/')
ds_i=2
tuning=ds[ds_i]
print(list(set(tuning.TSid)))
error = SWS().generate_error_signal(tuning,interpolation_name='Random Walk')
i=0
th=np.mean(error[i:i+5])+3*np.std(error[i:i+5])
for i in range(len(error)-14):
    #print(,np.percentile(error[i:i+10],97))
    plt.scatter(i,np.mean(error[i:i+5]))
    #print(np.where(error[i:i+10]>=th)[0])
    if len(np.where(error[i:i+5]>=th)[0])>0:
        #print("update",error[i:i+10])
        print("cut at:",i+5)
    else:
        th = np.mean(error[i:i + 5]) + 3 * np.std(error[i:i + 5])
plt.show()
sg=SegmentationBySWS().tuning(tuning,(tuning.TSid,tuning.label))
ds_i=1
for ds_i in [1,3,4,5,6,7,8,9]:
    tuning=ds[ds_i]
    s = TrajectorySegmentation()
    tsid = np.array(tuning.loc[:, ['TSid']].values.ravel())
    label = np.array(tuning.loc[:, ['label']].values.ravel())
    for i in range(len(tsid)-1):
        if tsid[i]!=tsid[i+1]:
            plt.axvline(x=i,c='r',alpha=0.2)
            #print(i)

    for k,v in tuning.groupby(['TSid']):
        print(v.shape)
        #error = SWS().generate_error_signal(v,interpolation_name='Random Walk',window_size=7)
        seg_id=sg.predict(trajectory=v,percentile=None)
        #print(k,list(set(seg_id)),seg_id,np.min(error),np.max(error),np.mean(error),len(error))
        from Trajlib2.core.traj_reconstruct import plot_df,reconstruct_traj_df
        #plot_df(v,v.TSid,seg_id)
        dfo=reconstruct_traj_df(v,density=0.00009,time_threshold=88000)
        plot_df(dfo, dfo.tsid, dfo.label,df2=v,df2_segid=seg_id,title="",
                path="/Users/owner/Trajlib2/Trajlib2/databases/fishing",traj_id=str(k))
        #plt.show()






#import matplotlib.pyplot as plt
#plt.plot(error)
#plt.ylim([0,10000])
#ws=13
#plt.show()
#error=np.array(error)
#norm = np.linalg.norm(error)
#norm = error / np.sqrt(np.sum(error**2))
#plt.plot(norm)
#plt.ylim([0,0.0008])
#plt.show()
#mv=np.convolve(error, np.ones(ws), 'valid') / ws
import hoggorm as ho
#for i in range(len(error)-2*ws):
#    j=i+ws
#    slice1=[mv[j-ws:j]]
#    slice2 = [mv[j :j + ws]]

#    rv_results =np.cov(slice1,slice2)
    #print(j,rv_results[1][0])
#plt.plot(mv)
from Trajlib2.core.traj_reconstruct import plot_df
#plot_df(tuning,tuning.TSid,tuning.TSid)
#plt.show()
