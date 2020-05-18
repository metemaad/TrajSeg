import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_df(col='Harmonic mean'):
    df=pd.DataFrame()
    f1=np.median

    xcl = []

    #df2 = pd.read_csv('Results_FishingDataset_SWS_C_WS_7.csv')
    #xcl.append(f1(df2['k_tsid']))
    #df = df.assign(base=df2['k_tsid'])
    df2 = pd.read_csv('Results_FishingDataset_SWS_C_WS_5.csv')
    xcl.append(f1(df2[col]))
    df = df.assign(SWS_C_ws5=df2[col])

    df2 = pd.read_csv('Results_FishingDataset_SWS_C_WS_7.csv')
    xcl.append(f1(df2[col]))
    df = df.assign(SWS_C_ws7=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_C_WS_9.csv')
    xcl.append(f1(df2[col]))
    df = df.assign(SWS_C_ws9=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_C_WS_11.csv')
    xcl.append(f1(df2[col]))
    df = df.assign(SWS_C_ws11=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_C_WS_11.csv')
    xcl.append(f1(df2[col]))
    df = df.assign(SWS_C_ws13=df2[col])

    df2 = pd.read_csv('Results_FishingDataset_SWS_L.csv')
    df = df.assign(SWS_L_WS3=df2[col])

    df2 = pd.read_csv('Results_FishingDataset_SWS_K.csv')
    df = df.assign(OWS_K_WS7=df2[col])

    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_5.csv')
    df = df.assign(SWS_RW_5=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_7.csv')
    df = df.assign(SWS_RW_7=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_9.csv')
    df = df.assign(SWS_RW_9=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_11.csv')
    df = df.assign(SWS_RW_11=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_13.csv')
    df = df.assign(SWS_RW_13=df2[col])

    #df2 = pd.read_csv('Results_FishingDataset_GRAST-UTS.csv')
    #dic = {'Harmonic mean': 'h', 'Purity': 'p', 'Coverage': 'c'}
    #df = df.assign(GRASP_UTS=df2[dic[col]] * 100)

    df2 = pd.read_csv('Results_FishingDataset_SPD.csv')
    df = df.assign(SPD=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_WKMeans.csv')
    df = df.assign(WKMeans=df2[col])

    df2 = pd.read_csv('Results_FishingDataset_CBSMoT.csv')

    df = df.assign(CBSMoT=df2[col])

    df2 = pd.read_csv('Results_FishingDataset_WSII.csv')

    df = df.assign(WSII=df2[col])

    df2 = pd.read_csv('Results_FishingDataset_SWS_LR_WS_5.csv')
    df = df.assign(SWS_LR_5=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_LR_WS_7.csv')
    df = df.assign(SWS_LR_7=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_LR_WS_9.csv')
    df = df.assign(SWS_LR_9=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_LR_WS_11.csv')
    df = df.assign(SWS_LR_11=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_LR_WS_13.csv')
    print(df2.columns)
    df = df.assign(SWS_LR_13=df2[col])


    print(df)
    return df,xcl


def plot(df,label,xcl):
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
    ax = sns.boxplot( data=df,orient="v")



    plt.ylabel(label)
    plt.xlabel("Algorithms")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis="x",direction="in", pad=-220)
    plt.tight_layout()
    plt.title("Fishing Dataset")
    #plt.ylim([0,100])








    plt.show()
#df2=pd.read_csv('Results_FishingDataset_WKMeans.csv')
#print(df2.columns)
# 'Purity', 'Coverage', 'Harmonic mean', 'best parameters',
#        'k_segment1', 'k_segment2', 'k_tsid', 'acc', 'kappa', 'pr', 're', 'j',
#        'p2', 'c2', 'h2', 'homo', 'comp', 'v_measure']
dfh,xcl=load_df('v_measure')


plot(dfh,'v_measure',xcl)
print(dfh.mean())

