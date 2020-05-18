import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def load_df(col='Harmonic mean'):
    df=pd.DataFrame()



    df2=pd.read_csv('Results_HurricanesDataset_SWS_C_WS_5.csv')
    df2=pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[5]*df2.shape[0])
    df2 = df2.assign(Algorithms=['Cubic'] * df2.shape[0])
    df=df2


    df2 = pd.read_csv('Results_HurricanesDataset_SWS_C_WS_7.csv')
    df2=pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[7] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Cubic'] * df2.shape[0])
    df = df.append(df2)



    df2 = pd.read_csv('Results_HurricanesDataset_SWS_C_WS_9.csv')
    df2=pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[9] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Cubic'] * df2.shape[0])
    df = df.append(df2)


    df2 = pd.read_csv('Results_HurricanesDataset_SWS_C_WS_11.csv')
    df2=pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[11] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Cubic'] * df2.shape[0])
    df = df.append(df2)


    df2 = pd.read_csv('Results_HurricanesDataset_SWS_C_WS_13.csv')
    df2=pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[13] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Cubic'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_RW_WS_5.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[5] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Random Walk'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_RW_WS_7.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[7] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Random Walk'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_RW_WS_9.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[9] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Random Walk'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_RW_WS_11.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[11] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Random Walk'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_RW_WS_13.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[13] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Random Walk'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_LR_WS_5.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[5] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Linear Regression'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_LR_WS_7.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[7] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Linear Regression'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_LR_WS_9.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[9] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Linear Regression'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_LR_WS_11.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[11] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Linear Regression'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('Results_HurricanesDataset_SWS_LR_WS_13.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(ws=[13] * df2.shape[0])
    df2 = df2.assign(Algorithms=['Linear Regression'] * df2.shape[0])
    df = df.append(df2)
#    sns.boxplot(x="ws", y="H_mean",
#                hue="Algorithms", palette=["m", "g", "y"],
#                data=df)
    return df,[]


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
    ax = sns.boxplot(x="ws", y='Harmonic mean',
                hue="Algorithms", data=df,orient="v",notch=False)
    #cx=df.loc[df.Algorithms=='Cubic',:].groupby(['ws']).sum().values/9.0


    plt.ylabel(label)
    plt.xlabel("Window Size")
    #plt.setp(ax.get_xticklabels(), rotation=45)
    #plt.setp(ax.get_xticklabels(), rotation=90)
    #ax.tick_params(axis="x",direction="in", pad=-220)
    plt.tight_layout()
    plt.title("Hurricanes Dataset")
    #plt.ylim([50,100])






    plt.savefig('SWS_Hurricanes_ws.png')

    plt.show()


dfh,xcl=load_df('Harmonic mean')

plot(dfh,'Harmonic mean',xcl)
for _,__ in dfh.groupby(['Algorithms']):

    for k,v in __.groupby('ws'):
        print(_,k,v.median().values,v.std().values)
    from minepy import MINE

    x = __['Harmonic mean'].values
    y = __['ws'].values
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    mine.compute_score(x, y)
    from scipy.spatial import distance
    import dcor
    print("dcor:",dcor.distance_stats(x, y))

    print("distance corrolation:",distance.correlation(x, y))
    print("Maximal Information Coefficient :", mine.mic())
    print(_,np.corrcoef(__['Harmonic mean'].values,__['ws'].values)[1][0])


