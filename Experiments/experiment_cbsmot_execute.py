import pandas as pd
import matplotlib.pyplot as plt
from Experiments.experiment_cbsmot import CBSMoTExperiment
from Trajlib2.databases import load_datasets

list_of_params = [{
    '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing2',
    '__algorithm__': 'CBSMoT_fishing2',
    '__dataset_name__': "FishingDataset",
    '__load_data_function__': load_datasets.load_data_fishing_data_fishing2,
    '__plotting__': False,
    '__seed__': 110,
    '__verbose__': False,
    '__tuning_parameters__': {'cbsmot_params': {'max_dist_param': [None], 'area_param': [0.01, 0.5, 0.3, 0.1, 0.7, 0.9],
                                                'min_time_param': [360, 3600, 60],
                                                'time_tolerance_param': [0], 'merge_tolerance_param': [0]}}},

    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': 'CBSMoT',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'cbsmot_params': {'max_dist_param': [None], 'area_param': [0.01, 0.5, 0.3, 0.1, 0.7, 0.9],
                              'min_time_param': [360, 3600, 60],
                              'time_tolerance_param': [0], 'merge_tolerance_param': [0]}}},
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__': 'CBSMoT',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'cbsmot_params': {'max_dist_param': [None], 'area_param': [0.01, 0.5, 0.3, 0.1, 0.7, 0.9],
                              'min_time_param': [360, 3600, 60],
                              'time_tolerance_param': [0], 'merge_tolerance_param': [0]}}}]
hh = []
for param_set in list_of_params:
    print(param_set)
    ex = CBSMoTExperiment(**param_set)
    h = ex.execute()
    print(hh, h)
    hh.append(h)

print(pd.DataFrame(hh))
pd.DataFrame(hh).boxplot()


plt.show()
