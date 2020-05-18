
from Trajlib2.databases import load_datasets
from Trajlib2.TrajectorySegmentation import TrajectorySegmentation
import Trajlib2.SegmentationEvaluation as segmentation_evaluation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import psutil
import random
def plot_segments(trajectory, segment_id,fold_id):
    plt.figure(figsize=(16, 16))
    # print(list(set(dp.TSid)))
    # seg_id=dp.TSid.values
    plt.title("fold "+str(fold_id))
    trajectory = trajectory.assign(seg_id=segment_id)
    trajectory = trajectory.sort_index()

    color = list(mcolors.TABLEAU_COLORS.values())  # CSS4_COLORS
    i = 0
    for segment_id in list(set(trajectory.seg_id)):
        sdf = trajectory.loc[trajectory.seg_id == segment_id, :]
        #for p in range(sdf.shape[0] - 2):
        #    plt.plot(sdf.lat[p:p + 2], sdf.lon[p:p + 2], c=color[i % len(color)], linewidth=1)
        #plt.plot(sdf.lat[-2:], sdf.lon[-2:], c=color[i % len(color)])
        plt.scatter(sdf.lat[:], sdf.lon[:], c=color[i % len(color)], s=20)
        i = i + 1
    # plt.xlim([3.7,4.5])
    # plt.ylim([-43.2,-42.8])
    plt.show()
random.seed(110)

process = psutil.Process(os.getpid())

track_memory=[]
start_time = time.process_time()

print(time.process_time()-start_time)
print(" Memory usage:",process.memory_info().rss/1000000,"MB")

track_memory.append(['start',time.process_time(),process.memory_info().rss/1000000])


print(pd.DataFrame(track_memory,columns=['milestone','time','memory']))
__Experiment_Name__="FishingDataset_WKMeans"


plot=False

track_memory.append(['loading data',time.process_time(),process.memory_info().rss/1000000])
ds = load_datasets.load_data_fishing_data(path='~/Trajlib2/Trajlib2/databases/fishing/')
track_memory.append(['data loaded',time.process_time(),process.memory_info().rss/1000000])

#for d in ds:
#    print("data shape:", d.shape)
listOfDatasets = set(range(len(ds)))
#print(listOfDatasets)

import matplotlib.colors as mcolors




