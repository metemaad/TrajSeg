from sklearn.model_selection import ParameterGrid

from Trajlib2.SegmentationAlgorithms.GRASP_UTS.GRASPUTS import GRASPUTS,GRASPUTSEvaluate
import pandas as pd
import time

if __name__ == "__main__":
    dfs = []
    dfs.append(pd.read_csv('databases/Hurricanes/h_d1.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d2.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d3.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d4.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d5.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d6.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d7.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d8.txt',sep=';'))
#    dfs.append(pd.read_csv('databases/Hurricanes/h_d9.txt',sep=';'))
    real_dfs = []
    for df in dfs:
        df.set_index(keys=['tid'],drop=False,inplace=True)
        tids = df['tid'].unique().tolist()
        for tid in tids:
            real_dfs.append(df.loc[df.tid ==tid])
    print(len(real_dfs))
    feature_names = ['direction_inference','speed_inference_m_s','wind']
    count=0
    alpha = [0.3]
    partitioning_factor =[0]
    max_iterations=[30]
    min_times = [6,12,18,24]
    jcs = [0.3,0.4]
    grid = {'alpha':alpha, 'partitioning_factor': partitioning_factor, 'max_iterations':max_iterations,'min_time':min_times,'jcs':jcs}
    parm_grid = ParameterGrid(grid)
    for p in parm_grid:
        grasputs = GRASPUTS(feature_names=feature_names,
                            lat_col='latitude', lon_col='longitude',
                            time_format='%Y-%m-%d %H:%M:%S',
                            min_time=p['min_time'],
                            join_coefficient=p['jcs'],
                            alpha=p['alpha'],
                            partitioning_factor=p['partitioning_factor'])
        print(p)
        for df in real_dfs:
            count+=1
            start_time = time.time()
            print("Trajectory", count)
            segs, cost = grasputs.segment(df)

            ground_truth = GRASPUTSEvaluate.get_ground_truth(df,label='label')

            prediction = GRASPUTSEvaluate.get_predicted(segs)
            print(prediction)
            print("Purity",GRASPUTSEvaluate.purity(ground_truth,prediction)[1])
            print("Coverage",GRASPUTSEvaluate.coverage(ground_truth,prediction)[1])
            print("Harmonic Mean",GRASPUTSEvaluate.harmonic_mean(ground_truth,prediction))
            print("--- %s seconds ---" % (time.time() - start_time))


