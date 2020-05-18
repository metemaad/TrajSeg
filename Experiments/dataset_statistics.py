from Trajlib2.databases import load_datasets


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def stat(ds,ws=13):
    test = pd.DataFrame()
    for k in range(len(ds)):
        test = test.append(ds[k])

        a = []

        for _ in set(ds[k]['TSid'].values):
            a.append(ds[k].loc[ds[k]['TSid'] == _, :].shape[0])
        a = np.array(a)
        print("Fold", k, ds[k].shape[0], len(set(ds[k]['TSid'].values)),
        100.0 * np.sum(a[a < ws]) / test.shape[0],np.sum(a[a < ws]))
        a = []
        for _ in set( ds[k]['TSid'].values):
            a.append( ds[k].loc[ ds[k]['TSid'] == _, :].shape[0])
        a = np.array(a)
        print("smallest and longest and mean:", np.min(a), np.max(a), np.mean(a))

    print("Number of trajectory points:",test.shape[0])
    print("Number of segments:",len(set(test['TSid'].values)))
    a=[]

    for _ in set(test['TSid'].values):
        a.append(test.loc[test['TSid']==_,:].shape[0])
    a=np.array(a)
    print("smallest and longest and mean:",np.min(a),np.max(a),np.mean(a))
    print("persentage of TP in short segments:",np.sum(a[a<13]),100.0*np.sum(a[a<7])/test.shape[0])
    ax = pd.DataFrame(a,columns=['tl']).plot.hist(bins=12, alpha=0.5)
    plt.axhline(y=ws,c='r',ls=':')
    plt.show()
#ds=load_datasets.load_data_fishing_data('~/Trajlib2/Trajlib2/databases/fishing')
#stat(ds)

#ds=load_datasets.load_data_hurricane_data('~/Trajlib2/Trajlib2/databases/hurricanes')
#stat(ds)


ds=load_datasets.load_data_geolife_data('~/Trajlib2/Trajlib2/databases/geolife2')
#stat(ds)

#ds=load_datasets.load_data_AIS_data()
#print(ds)
t=[]
from Trajlib2.SegmentationEvaluation import purity, coverage
p_u=0
p_o=0
c_u=0
c_o=0
for _ in ds:

    t.extend(list(np.array(np.diff(_.index)/1000000000).astype(float)))
    a = _.TSid
    #print(len(_.TSid))
    b=[0]*len(_.TSid)
    print("undersegmentation:")
    p_u=p_u+purity(a, b)[1]
    c_u = c_u + coverage(a, b)[1]
    print("p:",purity(a, b)[1])
    print("c:",coverage(a, b)[1])
    c=list(range(len(_.TSid)))
    print("oversegmentation:")
    print("p:",purity(a, c)[1])
    print("c:",coverage(a, c)[1])
    p_o = p_o + purity(a, c)[1]
    c_o = c_o + coverage(a, c)[1]


    #print(len(t))

print("under p:",p_u/10.,"c:",c_u/10.)
print("over p:",p_o/10.,"c:",c_o/10.)
print(len(t),"freq:",np.mean(t),np.percentile(t,75),np.std(t)," seconds", np.mean(t)/60,"mins", np.mean(t)/3600,"hrs")
print(np.mean(t)-np.percentile(t,50), np.mean(t)+np.percentile(t,50))
print(np.std(t),np.std(t)/1000)


