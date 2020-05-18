from sklearn.model_selection import ParameterGrid

from GRASPUTS import GRASPUTS,GRASPUTSEvaluate
import pandas as pd
import time

dfs = []
dfs.append(pd.read_csv('databases/Hurricanes/h_d1.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d2.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d3.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d4.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d5.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d6.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d7.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d8.txt',sep=';'))
dfs.append(pd.read_csv('databases/Hurricanes/h_d9.txt',sep=';'))

feature_names = ['direction_inference','speed_inference_m_s','wind']
count=0
alpha = [0.3]
partitioning_factor =[0]
max_iterations=[30]
min_times = [6,12,18,24]
jcs = [0.1,0.2,0.3,0.4]

grid = {'alpha':alpha, 'partitioning_factor': partitioning_factor, 'max_iterations':max_iterations,'min_time':min_times,'jcs':jcs}
parm_grid = ParameterGrid(grid)

def split_df(df,label='tid'):
    real_dfs = []
    df.set_index(keys=['tid'],drop=False,inplace=True)
    tids = df['tid'].unique().tolist()
    for tid in tids:
        real_dfs.append(df.loc[df.tid ==tid])
    return real_dfs

for i in range(0,len(dfs)):
    dataframes = split_df(dfs[i])
    parm_grid = ParameterGrid(grid)
    best_parameters = {}
    lowest_cost = float('inf')
    for p in parm_grid:
        grasputs = GRASPUTS(feature_names=feature_names,
                            lat_col='latitude', lon_col='longitude',
                            time_format='%Y-%m-%d %H:%M:%S',
                            min_time=p['min_time'],
                            join_coefficient=p['jcs'],
                            alpha=p['alpha'],
                            partitioning_factor=p['partitioning_factor'])
        total_cost = 0
        for df in dataframes:
            segs, cost = grasputs.segment(df)
            total_cost+=cost
        if total_cost<lowest_cost:
            lowest_cost = total_cost
            best_parameters = p
            #print(best_parameters)
    test_dfs = []
    print("Best Parameters:",best_parameters)
    for j in range(0, len(dfs)):
        if j != i:
            test_dfs+=split_df(dfs[j])
    grasputs = GRASPUTS(feature_names=feature_names,
                        lat_col='latitude', lon_col='longitude',
                        time_format='%Y-%m-%d %H:%M:%S',
                        min_time=best_parameters['min_time'],
                        join_coefficient=best_parameters['jcs'],
                        alpha=best_parameters['alpha'],
                        partitioning_factor=best_parameters['partitioning_factor'])
    total_purity = 0
    total_coverage = 0
    for df in test_dfs:
        count += 1
        start_time = time.time()
        #print(type(df))
        segs, cost = grasputs.segment(df)
        ground_truth = GRASPUTSEvaluate.get_ground_truth(df, label='label')
        prediction = GRASPUTSEvaluate.get_predicted(segs)
        total_purity += GRASPUTSEvaluate.purity(ground_truth, prediction)[1]
        total_coverage += GRASPUTSEvaluate.coverage(ground_truth, prediction)[1]
    avg_pur = total_purity/len(test_dfs)
    avg_cov = total_coverage/len(test_dfs)
    print("Purity",avg_pur)
    print("Coverage",avg_cov)
    print("Harmonic Mean", (2*avg_cov*avg_pur)/(avg_cov+avg_pur))
    print("*"*50)
    #print("--- %s seconds ---" % (time.time() - start_time))

########################################################################################################################

