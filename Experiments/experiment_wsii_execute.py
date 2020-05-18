import pandas as pd
import matplotlib.pyplot as plt
from Experiments.experiment_wsii import WSIIExperiment
from Trajlib2.databases import load_datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

list_of_params = [{
    '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing2',
    '__algorithm__': 'WSII_ws5_fishing2',
    '__dataset_name__': "FishingDataset",
    '__load_data_function__': load_datasets.load_data_fishing_data_fishing2,
    '__plotting__': False,
    '__seed__': 110,
    '__verbose__': False,
    '__tuning_parameters__': {
         'wsii_params': {'window_size': [7], 'majority_vote_degree': [0.9], 'binary_classifier': [RandomForestClassifier(n_estimators=100)]}}}
    ,
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': 'WSII_ws5',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'window_size': [5], 'majority_vote_degree': [0.9], 'binary_classifier': [None]}}},

    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__': 'WSII_ws5',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
             'wsii_params': {'window_size': [5], 'majority_vote_degree': [0.9], 'binary_classifier': [None]}}}]
hh = []
for param_set in list_of_params:
    print(param_set)
    ex = WSIIExperiment(**param_set)
    h = ex.execute()
    print(hh, h)
    hh.append(h)

print(pd.DataFrame(hh))
pd.DataFrame(hh).boxplot()

plt.show()
