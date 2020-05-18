import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from Experiments.experiment_wsii import WSIIExperiment
from Trajlib2.databases import load_datasets
majority_vote_degree=0.9

list_of_params = [{
    '__datafile__': '~/Trajlib2/Trajlib2/databases/fishing/',
    '__algorithm__': 'MTS_WSII_LR_'+str(majority_vote_degree)+'NN',
    '__dataset_name__': "FishingDataset",
    '__load_data_function__': load_datasets.load_data_fishing_data_mts,
    '__plotting__': False,
    '__seed__': 110,
    '__verbose__': False,
    '__tuning_parameters__': {
        'wsii_params': {'kernel':["linear"],'window_size': [7], 'majority_vote_degree': [majority_vote_degree],
                        'binary_classifier': [RandomForestClassifier()]}}}]
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
