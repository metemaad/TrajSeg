import pandas as pd
def load_df(col='Harmonic mean'):
    df = pd.DataFrame()

    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_WSII_0.6RF.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(MajorityVote=['0.6'] * df2.shape[0])
    df = df2.copy()

    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_WSII_0.9RF.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(MajorityVote=['0.9'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_WSII_0.6RF.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(MajorityVote=['0.6'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_WSII_0.9RF.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(MajorityVote=['0.9'] * df2.shape[0])
    df = df.append(df2)
    df2 = pd.read_csv('GeolifeDataset/Results_GeolifeDataset_WSII_0.6RF.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Geolife'] * df2.shape[0])
    df2 = df2.assign(MajorityVote=['0.6'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('GeolifeDataset/Results_GeolifeDataset_WSII_0.9RF.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Geolife'] * df2.shape[0])
    df2 = df2.assign(MajorityVote=['0.9'] * df2.shape[0])
    df = df.append(df2)



    return df

df=load_df()

import seaborn as sns
import matplotlib.pyplot as plt
SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
sns.set_style("white")
fig=plt.figure(figsize=(16,12))

ax = sns.boxplot( x="DataSet", y="Harmonic mean",
                hue="MajorityVote", palette=["m", "g", "y"],
                data=df)
plt.savefig('MajorityVote_wsii.png')
plt.show()


for k,v in df.groupby(['DataSet','MajorityVote']):
    print(k[0],k[1],v.mean(),v.std())

import numpy as np
freq=[1363.8514171557,6343.528957528957,29923.636363636364]
label=['Geolife','Fishing','Hurricanes']
m06=np.array([91.76202,67.315711,71.375195])
m09=np.array([92.414754,68.850195,82.944491])
print(np.corrcoef(freq,m06)[1][0])
print(np.corrcoef(freq,m09)[1][0])

from scipy.stats import ranksums as kruskal
from scipy.stats import iqr
dfh=df
for _, __ in dfh.groupby(['DataSet']):
    print(_)
    for k1, v1 in __.groupby('MajorityVote'):
        print(k1,'mdn:%0.2f' % v1['Harmonic mean'].median(),
                  'iqr:%0.2f' % iqr(v1['Harmonic mean']))
        for k, v in __.groupby('MajorityVote'):
            p, s = kruskal(v1['Harmonic mean'], v['Harmonic mean'])
            print(_, k1, k, 'mdn:%0.2f' % v['Harmonic mean'].median(),
                  'iqr:%0.2f' % iqr(v['Harmonic mean']), 's:%0.5f' % p, 'p:%0.5f' % s)
