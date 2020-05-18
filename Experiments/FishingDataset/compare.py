import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_df(col='Harmonic mean'):
    df=pd.DataFrame()
    f1=np.median

    xcl=[]
    df2=pd.read_csv('Results_FishingDataset_SWS_C_WS_5.csv')
    xcl.append(f1(df2[col]))
    df=df.assign(SWS_C_ws5=df2[col])

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



    df2=pd.read_csv('Results_FishingDataset_SWS_L.csv')
    df=df.assign(SWS_L_WS3=df2[col])

    df2=pd.read_csv('Results_FishingDataset_SWS_K.csv')
    df=df.assign(OWS_K_WS7=df2[col])

    df2=pd.read_csv('Results_FishingDataset_SWS_RW_WS_5.csv')
    df=df.assign(SWS_RW_5=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_7.csv')
    df = df.assign(SWS_RW_7=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_9.csv')
    df = df.assign(SWS_RW_9=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_11.csv')
    df = df.assign(SWS_RW_11=df2[col])
    df2 = pd.read_csv('Results_FishingDataset_SWS_RW_WS_13.csv')
    df = df.assign(SWS_RW_13=df2[col])

    df2=pd.read_csv('Results_FishingDataset_GRASPUTS.csv')
    #dic={'Harmonic mean':'h','Purity':'p','Coverage':'c'}
    #df=df.assign(GRASP_UTS=df2[dic[col]]*100)
    df = df.assign(GRASP_UTS=df2[col])


    df2=pd.read_csv('Results_FishingDataset_SPD.csv')
    df=df.assign(SPD=df2[col])
    df2=pd.read_csv('Results_FishingDataset_WKMeans.csv')
    df=df.assign(WKMeans=df2[col])

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
    df = df.assign(SWS_LR_13=df2[col])



    #print(df)
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
    ax.axvspan(-0.5,4.5,alpha=0.2,color='g',fill='+',lw=4)
    #ax.axvspan(0.5,1.5,alpha=0.2,color='w')
    #ax.axvspan(1.5,2.5,alpha=0.2,color='w')
    #ax.axvspan(2.5,3.5,alpha=0.2,color='w')

    #ax.axvspan(3.5,4.5,alpha=0.2,color='w')
    ax.axvspan(4.5,5.5,alpha=0.2,color='k',fill='+',lw=4)
    ax.axvspan(5.5,6.5,alpha=0.2,color='r',fill='+',lw=4)
    ax.axvspan(6.5, 11.5, alpha=0.2, color='y',fill='/',lw=4)
    #ax.axvspan(7.5, 8.5, alpha=0.2, color='w')
    #ax.axvspan(8.5, 9.5, alpha=0.2, color='w')
    #ax.axvspan(9.5, 10.5, alpha=0.2, color='w')
    #ax.axvspan(10.5, 11.5, alpha=0.2, color='w')
    ax.axvspan(11.5, 12.5, alpha=0.2, color='pink')
    ax.axvspan(12.5, 13.5, alpha=0.2, color='b')
    ax.axvspan(13.5, 15.5, alpha=0.2, color='w')
    #ax.axvspan(14.5, 15.5, alpha=0.2, color='w')
    ax.plot([-.5,0,1,2,3,4,4.5],np.array([xcl[0]]+xcl+[xcl[-1]]),c='w',linewidth=5,alpha=0.8)
    ax.plot([-.5,0,1,2,3,4,4.5],np.array([xcl[0]]+xcl+[xcl[-1]]), c='darkblue', linewidth=3,alpha=0.8)
    aa=np.array([[df.mean()[0]]+[df.mean()[0]]+[df.mean()[1]]+
             [df.mean()[2]]+[df.mean()[3]]+[df.mean()[4]]+
             [df.mean()[4]]])-np.array([[df.std()[0]]+[df.std()[0]]+[df.std()[1]]+
             [df.std()[2]]+[df.std()[3]]+[df.std()[4]]+
             [df.std()[4]]])
    aam = np.array([[df.mean()[0]] + [df.mean()[0]] + [df.mean()[1]] +
                   [df.mean()[2]] + [df.mean()[3]] + [df.mean()[4]] +
                   [df.mean()[4]]]) + np.array([[df.std()[0]] + [df.std()[0]] + [df.std()[1]] +
                                                [df.std()[2]] + [df.std()[3]] + [df.std()[4]] +
                                                [df.std()[4]]])
    ax.fill_between([-.5, 0, 1, 2, 3, 4, 4.5],aa.ravel(),aam.ravel(),alpha=0.2)
    ax.plot([-.5, 0, 1, 2, 3, 4, 4.5],aa.ravel()
            , c='darkblue', linewidth=3, alpha=0.8)
    ax.plot([-.5, 0, 1, 2, 3, 4, 4.5], aam.ravel()
            , c='darkblue', linewidth=3, alpha=0.8)
    ax.text(0,45, "SWS Cubic", fontsize=42)
    ax.text(7, 45, "SWS Random Walk", fontsize=32)

    aa = np.array([[df.mean()[7]] +[df.mean()[7]] + [df.mean()[8]] + [df.mean()[9]] +
                   [df.mean()[10]] + [df.mean()[11]]+ [df.mean()[11]] ]) -\
         np.array([[df.std()[7]] +[df.std()[7]] + [df.std()[8]] + [df.std()[9]] +
                                                [df.std()[10]] + [df.std()[11]] + [df.std()[11]] ])
    aam = np.array([[df.mean()[7]] +[df.mean()[7]] + [df.mean()[8]] + [df.mean()[9]] +
                   [df.mean()[10]] + [df.mean()[11]]+ [df.mean()[11]] ])+\
         np.array([[df.std()[7]] +[df.std()[7]] + [df.std()[8]] + [df.std()[9]] +
                                                [df.std()[10]] + [df.std()[11]] + [df.std()[11]] ])
    ax.fill_between([6.5, 7, 8, 9, 10, 11, 11.5], aa.ravel(), aam.ravel(), alpha=0.2)
    ax.plot([6.5, 7, 8, 9, 10, 11, 11.5], aa.ravel()
            , c='darkblue', linewidth=3, alpha=0.8)
    ax.plot([6.5, 7, 8, 9, 10, 11, 11.5], aam.ravel() , c='darkblue', linewidth=3, alpha=0.8)

    #bar = pd.DataFrame().assign(s=np.array(df.std())+60)
    #bar = bar.reindex()
    #ax=bar.plot.bar(ax=ax,legend=None)
    #ax.set_xticklabels(df.columns)

    plt.ylabel(label)
    plt.xlabel("Algorithms")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis="x",direction="in", pad=-220)
    plt.tight_layout()
    plt.title("Fishing Dataset")
    plt.ylim([0,100])








    plt.show()


dfh,xcl=load_df('Harmonic mean')

plot(dfh,'Harmonic mean',xcl)
print(dfh.mean())
from scipy import stats
t2, p2 = stats.ttest_ind(dfh['SWS_L_WS3'],dfh['CBSMoT'])
print("t = " + str(t2))
print("p = " + str(p2))

t2, p2 = stats.ttest_ind(dfh['SWS_C_ws13'],dfh['CBSMoT'])
print("t = " + str(t2))
print("p = " + str(p2))

t2, p2 = stats.ttest_ind(dfh['OWS_K_WS7'],dfh['SPD'])
print("t = " + str(t2))
print("p = " + str(p2))


t2, p2 = stats.ttest_ind(dfh['SWS_C_ws13'],dfh['SWS_C_ws5'])
print("t = " + str(t2))
print("p = " + str(p2))

dfh,xls=load_df('Purity')
plot(dfh,'Purity',xls)

dfh,xls=load_df('Coverage')
plot(dfh,'Coverage',xls)
#print(dfh.mean())