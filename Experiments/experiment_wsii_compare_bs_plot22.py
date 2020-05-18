import pandas as pd
strr='KI'
df = pd.DataFrame()
df2 = pd.read_csv('FishingDataset/Results_FishingDataset_'+strr+'WSII__0.9DT.csv')
print(df2.columns)
df = df.assign(H_mean=df2['Harmonic mean'])
df = df.assign(Dataset=['Fishing'] * df2.shape[0])
df = df.assign(b_cls=['DT'] * df2.shape[0])


def add_data(df, file='FishingDataset/Results_FishingDataset_'+strr+'WSII__0.9RF.csv'
             , dataset='Fishing', cls='RF'):
    df1 = pd.DataFrame()
    dfr = pd.read_csv(file)
    df1 = df1.assign(H_mean=dfr['Harmonic mean'])
    df1 = df1.assign(Dataset=[dataset] * dfr.shape[0])
    df1 = df1.assign(b_cls=[cls] * dfr.shape[0])
    df = df.append(df1, ignore_index=True)
    return df.copy()


df = add_data(df, 'FishingDataset/Results_FishingDataset_'+strr+'WSII__0.9RF.csv', 'Fishing', 'RF')
df = add_data(df, 'FishingDataset/Results_FishingDataset_'+strr+'WSII__0.9NN.csv', 'Fishing', 'NN')
df = add_data(df, 'FishingDataset/Results_FishingDataset_'+strr+'WSII__0.9NB.csv', 'Fishing', 'NB')

df = add_data(df, 'HurricanesDataset/Results_HurricanesDataset_'+strr+'WSII__0.9DT.csv', 'Hurricanes', 'DT')
df = add_data(df, 'HurricanesDataset/Results_HurricanesDataset_'+strr+'WSII__0.9RF.csv', 'Hurricanes', 'RF')
df = add_data(df, 'HurricanesDataset/Results_HurricanesDataset_'+strr+'WSII__0.9NN.csv', 'Hurricanes', 'NN')
df = add_data(df, 'HurricanesDataset/Results_HurricanesDataset_'+strr+'WSII__0.9NB.csv', 'Hurricanes', 'NB')

df = add_data(df, 'GeolifeDataset/Results_GeolifeDataset_'+strr+'WSII__0.9DT.csv', 'Geolife', 'DT')
df = add_data(df, 'GeolifeDataset/Results_GeolifeDataset_'+strr+'WSII__0.9RF.csv', 'Geolife', 'RF')
df = add_data(df, 'GeolifeDataset/Results_GeolifeDataset_'+strr+'WSII__0.9NN.csv', 'Geolife', 'NN')
df = add_data(df, 'GeolifeDataset/Results_GeolifeDataset_'+strr+'WSII__0.9NB.csv', 'Geolife', 'NB')

print(df)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", palette="pastel")
sns.boxplot(x="b_cls", y="H_mean",
            hue="Dataset", palette=["m", "g", "y"],
            data=df)
sns.despine(offset=10, trim=True)
plt.savefig('wsii_compare_bc.png')
plt.interactive(False)
plt.xlabel("Binary Classifier")
plt.ylabel("Harmonic means")
plt.show()

for k,v in df.groupby(['b_cls','Dataset']):
    print(k[0],k[1],v.mean(),v.std())
from scipy.stats import ranksums as kruskal
from scipy.stats import iqr
dfh=df
for _, __ in dfh.groupby(['Dataset']):
    print(_)
    for k1, v1 in __.groupby('b_cls'):
        print(k1,'mdn:%0.2f' % v1['H_mean'].median(),
                  'iqr:%0.2f' % iqr(v1['H_mean']))
        for k, v in __.groupby('b_cls'):
            p, s = kruskal(v1['H_mean'], v['H_mean'])
            print(_, k1, k, 'mdn:%0.2f' % v['H_mean'].median(),
                  'iqr:%0.2f' % iqr(v['H_mean']), 's:%0.5f' % p, 'p:%0.5f' % s)

for _, __ in dfh.groupby(['Dataset']):
    print(_)
    for k1, v1 in __.groupby('b_cls'):
        print(k1,'mdn:%0.2f' % v1['H_mean'].median(),
                  'iqr:%0.2f' % iqr(v1['H_mean']))