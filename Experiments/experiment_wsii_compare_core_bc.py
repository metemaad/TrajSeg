import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from Experiments.experiment_wsii import WSIIExperiment
from Trajlib2.databases import load_datasets
majority_vote_degree=0.9
#'cubic', 'linear', 'kinematic'
kernels=["kinematic"]
kernel='KI'

list_of_params = [{
    '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing/',
    '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'DT',
    '__dataset_name__': "FishingDataset",
    '__load_data_function__': load_datasets.load_data_fishing_data,
    '__plotting__': False,
    '__seed__': 110,
    '__verbose__': False,
    '__tuning_parameters__': {
        'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                        'binary_classifier': [DecisionTreeClassifier(max_depth=10)]}}},
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing/',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'RF',
        '__dataset_name__': "FishingDataset",
        '__load_data_function__': load_datasets.load_data_fishing_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [RandomForestClassifier(n_estimators=100)]}}},
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing/',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'NN',
        '__dataset_name__': "FishingDataset",
        '__load_data_function__': load_datasets.load_data_fishing_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree], 'binary_classifier': [
                MLPClassifier(hidden_layer_sizes=(7,), max_iter=3000, alpha=1e-3,
                              learning_rate_init=.4)]}}},
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing/',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'NB',
        '__dataset_name__': "FishingDataset",
        '__load_data_function__': load_datasets.load_data_fishing_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree], 'binary_classifier': [GaussianNB()]}}},
    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'DT',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [DecisionTreeClassifier(max_depth=10)]}}},
{
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'RF',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [RandomForestClassifier(n_estimators=100)]}}},
{
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'NN',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [MLPClassifier(hidden_layer_sizes=(7,), max_iter=100, alpha=1e-3,
                              learning_rate_init=.4)]}}},
{
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Hurricanes',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'NB',
        '__dataset_name__': "HurricanesDataset",
        '__load_data_function__': load_datasets.load_data_hurricane_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [GaussianNB()]}}},

    {
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'DT',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [DecisionTreeClassifier(max_depth=10)]}}},
{
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__':kernel+ 'WSII__'+str(majority_vote_degree)+'RF',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [RandomForestClassifier(n_estimators=100)]}}},
{
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'NN',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                            'binary_classifier': [MLPClassifier(hidden_layer_sizes=(7,), max_iter=100, alpha=1e-3,
                              learning_rate_init=.4)]}}},
{
        '__datafile__': '~/Trajlib2/Trajlib2/databases/Geolife2',
        '__algorithm__': kernel+'WSII__'+str(majority_vote_degree)+'NB',
        '__dataset_name__': "GeolifeDataset",
        '__load_data_function__': load_datasets.load_data_geolife_data,
        '__plotting__': False,
        '__seed__': 110,
        '__verbose__': False,
        '__tuning_parameters__': {
            'wsii_params': {'kernel':kernels,'window_size': [7], 'majority_vote_degree': [0.9],
                            'binary_classifier': [GaussianNB()]}}}]
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
