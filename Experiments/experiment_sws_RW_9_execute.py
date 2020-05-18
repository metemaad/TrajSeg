import pandas as pd
import matplotlib.pyplot as plt
from Experiments.experiment_sws import SWSExperiment
from Trajlib2.databases import load_datasets

list_of_params = [{
    '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing/',
    '__algorithm__': 'SWS_RW_WS_9',
    '__dataset_name__': "FishingDataset",
    '__load_data_function__': load_datasets.load_data_fishing_data,
    '__plotting__': False,
    '__seed__': 110,
    '__verbose__': False,
    '__tuning_parameters__': {
        'sws_params': {'window_size_param': [9], 'interpolation_kernel_param': ['Random Walk'], 'epsilon_param': [None]}}}
    ,
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': 'SWS_RW_WS_9',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'sws_params': {'window_size_param': [9], 'interpolation_kernel_param': ['Random Walk'], 'epsilon_param': [None]}}},
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__': 'SWS_RW_WS_9',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'sws_params': {'window_size_param': [9], 'interpolation_kernel_param': ['Random Walk'], 'epsilon_param': [None]}}}]
hh = []
for param_set in list_of_params:
    print(param_set)
    ex = SWSExperiment(**param_set)
    h = ex.execute()
    print(hh, h)
    hh.append(h)

print(pd.DataFrame(hh))
pd.DataFrame(hh).boxplot()

plt.show()
