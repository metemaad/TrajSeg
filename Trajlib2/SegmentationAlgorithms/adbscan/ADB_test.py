from Trajlib2.SegmentationAlgorithms.adbscan.ADBSCAN import ADBSCAN
import matplotlib.pyplot as plt

from Trajlib2.SegmentationAlgorithms.adbscan.Trajectory_Point import TrajectoryPoint

file = "/Users/owner/Trajlib2/Trajlib2/databases/fishing/fv_d2.txt"


def file_stream():
    i=0
    for line in open(file).read().splitlines():
        i=i+1
        #if i<2000:
        yield line

print("start adb")
adbscan = ADBSCAN(window_size=13,sensitivity=27,min_points=2,
                  speed_condition=False,bearing_cond=False)#,min_points=5,max_points=50)
i=1
for _ in file_stream():
   # print("new data point")
    tp=None
    try:
        #r,lid,t_user_id,collected_time,latitude,longitude,altitude,transportation_mode,geometry
        data = _.split(';')
        #db/geolife/geolife_w_features_1.csv: tp = TrajectoryPoint(lat=data[4], lon=data[5], time_date=data[0],label= data[2],mode= data[7],trajectory_id=0)
        tp = TrajectoryPoint(lat=data[3], lon=data[4], time_date=data[5],label= data[0],mode= data[9],trajectory_id=0)
        # for fishing2
        #tp = TrajectoryPoint(lat=data[1], lon=data[2], time_date=data[5], label=data[3], mode=data[4], trajectory_id=0)
        #print("read:[",data[4], data[5],"]","total:[",str(i)+"]")
        i=i+1

    except:
        print("error reading file")
        continue
 #   try:
    adbscan.cluster_trajectory_point(tp)
       # print("current cluster:",adbscan.current_cluster)
 #   except Exception as e:
  #      print("error clustering:",e)
from Trajlib2.SegmentationEvaluation import purity,coverage,harmonic_mean

adbscan.all_eps()
ground_truth=adbscan.get_segment_id()
labels=adbscan.get_ground_truth()
trans_mod=adbscan.get_ground_mode()
segment_id=adbscan.get_segment_id()
print(labels)
print(trans_mod)
print(segment_id)
print(purity(ground_truth_label=trans_mod, generated_segment_id=segment_id))
print(coverage(ground_truth_segment_id=labels, generated_segment_id=segment_id))
print(harmonic_mean(segments=segment_id, tsid=labels, label=trans_mod))
from Trajlib2.core.traj_reconstruct import plot_df
import pandas as pd
from Trajlib2.databases import load_datasets
ds=load_datasets.load_data_fishing_data("/Users/owner/Trajlib2/Trajlib2/databases/fishing")
print(ds[1].shape,len(labels))
plot_df(ds[1], labels[1:], segment_id[1:])

