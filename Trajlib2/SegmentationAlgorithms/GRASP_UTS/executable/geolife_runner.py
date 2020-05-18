from sklearn.model_selection import ParameterGrid

from GRASPUTS import GRASPUTS,GRASPUTSEvaluate
import pandas as pd
import time

if __name__ == "__main__":
    dfs = []
    dfs.append(pd.read_csv('databases/geolife/geolife_w_features.csv'))
    print(dfs)
    feature_names = ['speed','bearing']
    count=0
    alpha = [0.3]
    partitioning_factor =[0]
    max_iterations=[30]
    min_times = [6,12,18,24]
    jcs = [0.3,0.4]
    grid = {'alpha':alpha, 'partitioning_factor': partitioning_factor, 'max_iterations':max_iterations,'min_time':min_times,'jcs':jcs}
    parm_grid = ParameterGrid(grid)
    df = dfs[0]

    dfs[0] = df.iloc[:1000]
    for p in parm_grid:
        grasputs = GRASPUTS(feature_names=feature_names,
                            lat_col='lat', lon_col='lon',
                            time_format='%Y-%m-%d %H:%M:%S',
                            min_time=p['min_time'],
                            join_coefficient=p['jcs'],
                            alpha=p['alpha'],
                            partitioning_factor=p['partitioning_factor'])
        print(p)
        for df in dfs:
            count+=1
            start_time = time.time()
            print("Trajectory", count)
            segs, cost = grasputs.segment(df)
            ground_truth = GRASPUTSEvaluate.get_ground_truth(df,label='transportation_mode')
            prediction = GRASPUTSEvaluate.get_predicted(segs)
            print("Purity",GRASPUTSEvaluate.purity(ground_truth,prediction)[1])
            print("Coverage",GRASPUTSEvaluate.coverage(ground_truth,prediction)[1])
            print("Harmonic Mean",GRASPUTSEvaluate.harmonic_mean(ground_truth,prediction))
            print("--- %s seconds ---" % (time.time() - start_time))


