import pandas as pd
import matplotlib.pyplot as plt
from Experiments.experiment_grasputs import GRASPUTSExperiment
from Trajlib2.databases import load_datasets

list_of_params = [{
    '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing/',
    '__algorithm__': 'GRASPUTS',
    '__dataset_name__': "FishingDataset",
    '__load_data_function__': load_datasets.load_data_fishing_data,
    '__plotting__': False,
    '__seed__': 110,
    '__verbose__': False,
    '__tuning_parameters__': {'grasputs_params': {
        'alpha': [0.3,0.5,0.7],
                               'partitioning_factor': [0],
                               'max_iterations': [10,20,30],
                               'min_time': [6,60,360],
                               'jcs': [0.1,0.3,0.7]}

                              }}
    ,
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': 'GRASPUTS',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__':{'grasputs_params': {
        'alpha': [0.3,0.5,0.7],
                               'partitioning_factor': [0],
                               'max_iterations': [10,20,30],
                               'min_time': [6,60,360],
                               'jcs': [0.1,0.3,0.7]}

                              }
    },
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__': 'GRASPUTS',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {'grasputs_params': {
        'alpha': [0.3,0.5,0.7],
                               'partitioning_factor': [0],
                               'max_iterations': [10,20,30],
                               'min_time': [6,60,360],
                               'jcs': [0.1,0.3,0.7]}

                              }}]
hh = []
for param_set in list_of_params:
    print(param_set)
    ex = GRASPUTSExperiment(**param_set)
    h = ex.execute()
    print(hh, h)
    hh.append(h)

print(pd.DataFrame(hh))
pd.DataFrame(hh).boxplot()


plt.show()