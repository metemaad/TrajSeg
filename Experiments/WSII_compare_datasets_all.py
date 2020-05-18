import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_df(col='Harmonic mean'):
    df = pd.DataFrame()

    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_SWS_LR_WS_13.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['SWS'] * df2.shape[0])
    df = df2

    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_CBSMoT.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['CBSMoT'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_GRASPUTS.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['GRASP-UTS'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_SPD.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['SPD'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_WKMeans.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['WKMeans'] * df2.shape[0])
    df = df.append(df2)



    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_SWS_C_WS_7.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['SWS'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_CBSMoT.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['CBSMoT'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_GRASPUTS.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['GRASP-UTS'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_SPD.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['SPD'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_WKMeans.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['WKMeans'] * df2.shape[0])
    df = df.append(df2)


    df2 = pd.read_csv('GeolifeDataset/Results_GeolifeDataset_SWS_LR_WS_11.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Geolife'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['SWS'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('GeolifeDataset/Results_GeolifeDataset_CBSMoT.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Geolife'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['CBSMoT'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('GeolifeDataset/Results_GeolifeDataset_SPD.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Geolife'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['SPD'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('GeolifeDataset/Results_GeolifeDataset_WKMeans.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Geolife'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['WKMeans'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('GeolifeDataset/Results_GeolifeDataset_KIWSII__0.9RF.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Geolife'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['WSII'] * df2.shape[0])
    df = df.append(df2)

    df2 = pd.read_csv('HurricanesDataset/Results_HurricanesDataset_LIWSII__0.9NB.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Hurricanes'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['WSII'] * df2.shape[0])
    df = df.append(df2)
    df2 = pd.read_csv('FishingDataset/Results_FishingDataset_LIWSII__0.9NN.csv')
    df2 = pd.DataFrame(df2[col].copy())
    df2 = df2.assign(DataSet=['Fishing'] * df2.shape[0])
    df2 = df2.assign(Algorithms=['WSII'] * df2.shape[0])
    df = df.append(df2)

    #    sns.boxplot(x="ws", y="H_mean",
    #                hue="Algorithms", palette=["m", "g", "y"],
    #                data=df)
    return df, []


def plot(df, label, xcl):
    import seaborn as sns

    #    .plot(x, y ** 2)
    #    axs[1].plot(x, 0.3 * y, 'o')
    #    axs[2].plot(x, y, '+')

    # Hide x labels and tick labels for all but bottom plot.
    #

    SMALL_SIZE = 24
    MEDIUM_SIZE = 26
    BIGGER_SIZE = 28

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    sns.set_style("white")
    fig, axs = plt.subplots(2, figsize=(16, 12), sharex=True, sharey=False, gridspec_kw={'hspace': 0.1})

    axs[0] = sns.boxplot(ax=axs[0], x="DataSet", y='Harmonic mean',
                         hue="Algorithms", data=df, orient="v", notch=False)
    axs[0].set_ylim([60, 100])
    axs[0].set_ylabel('')

    axs[1] = sns.boxplot(ax=axs[1], x="DataSet", y='Harmonic mean',
                         hue="Algorithms", data=df, orient="v", notch=False)

    axs[1].set_ylabel('.                                              Harmonic mean')
    axs[1].set_ylim([20, 45])
    axs[0].legend().set_visible(False)
    #axs[1].axvspan(xmin=-0.5, xmax=0.5, color='lightblue', alpha=0.2)
    #axs[1].axvspan(xmin=0.5, xmax=1.5, color='w', alpha=0.2)
    #axs[1].axvspan(xmin=1.5, xmax=2.5, color='lightblue', alpha=0.2)

    #axs[0].axvspan(xmin=-0.5, xmax=0.5, color='lightblue', alpha=0.2)
    #axs[0].axvspan(xmin=0.5, xmax=1.5, color='w', alpha=0.2)
    #axs[0].axvspan(xmin=1.5, xmax=2.5, color='lightblue', alpha=0.2)
    axs[1].legend(loc=10, bbox_to_anchor=(0.52, 1.05),
                  ncol=5, fancybox=True, shadow=True, borderpad=0.55)
    #axs[0].axvspan(xmin=-0.40, xmax=0.06, color='g', alpha=0.2)  # swsfishing
    #axs[1].axvspan(xmin=-0.40, xmax=0.06, color='g', alpha=0.2)  # swsfishing
    # axs[0].axhline(y=91.56283801,xmin=0, xmax=0.15,color='b',linestyle=':',)
    # axs[0].axhline(y=91.89862074, xmin=0, xmax=0.20, color='orange', linestyle=':', )

    # axs[0].axvspan(xmin=-0.25, xmax=-0.08, color='g', alpha=0.2)#cbsmot fishing

    #axs[0].axvspan(xmin=0.6, xmax=0.9, color='g', alpha=0.2)  # cbsmot hurr
    #axs[1].axvspan(xmin=0.6, xmax=0.9, color='g', alpha=0.2)  # cbsmot hurr

    # axs[0].axhline(y=93.18962886, xmin=0, xmax=0.70, color='b', linestyle=':', )
    # axs[0].axhline(y=91.65718265, xmin=0, xmax=0.84, color='r', linestyle=':', )
    # axs[0].axhline(y=91.53922388, xmin=0, xmax=0.74, color='orange', linestyle=':', )

    #axs[0].axvspan(xmin=1.6, xmax=1.9, color='g', alpha=0.2)  # cbsmot geolife
    #axs[0].axvspan(xmin=2.09, xmax=2.24, color='g', alpha=0.2)  # cbsmot geolife
    #axs[1].axvspan(xmin=1.6, xmax=1.9, color='g', alpha=0.2)  # cbsmot geolife
    #axs[1].axvspan(xmin=2.09, xmax=2.24, color='g', alpha=0.2)  # cbsmot geolife
    # axs[0].ylabel(label)
    axs[1].set_xlabel("DataSets")
    # plt.setp(ax.get_xticklabels(), rotation=45)
    # plt.setp(ax.get_xticklabels(), rotation=90)
    # ax.tick_params(axis="x",direction="in", pad=-220)

    fig.suptitle("Compare WS-II with other algorithms on three Datasets")
    # axs[0].ylim([50,100])
    # import matplotlib.patches as patches
    # rect = patches.Rectangle((0.25, 0.75), 1, 1, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    # axs[0].add_patch(rect)

    for ax in axs:
        ax.label_outer()

    plt.savefig('WSII_compare_other_h.png')

    plt.show()


dfh, xcl = load_df('Harmonic mean')

plot(dfh, 'Harmonic mean', xcl)
for _, __ in dfh.groupby(['DataSet']):
    best = 0
    abest = [0, []]
    for k, v in __.groupby('Algorithms'):
        print(_, k, v.mean().values, v.std().values)
        if v.mean().values[0] > best:
            best = v.mean().values[0]
            abest = (k, v['Harmonic mean'])
    for k, v in __.groupby('Algorithms'):
        # print(abest[0])
        if k != abest[0]:
            from scipy import stats

            s, p = stats.ttest_ind(abest[1], v['Harmonic mean'].values)
            if (p <= 0.05):
                print("*T-Test:", abest[0], k, s, p)
            else:
                print("T-Test:", abest[0], k, s, p)
from scipy.stats import ranksums as kruskal
from scipy.stats import iqr
plot(dfh, 'Harmonic mean', xcl)
for _, __ in dfh.groupby(['DataSet']):
    print(_)
    for k1, v1 in __.groupby('Algorithms'):
        print(k1,'mdn:%0.2f' % v1['Harmonic mean'].median(),
                  'iqr:%0.2f' % iqr(v1['Harmonic mean']))
        for k, v in __.groupby('Algorithms'):
            s,p = kruskal(v1['Harmonic mean'], v['Harmonic mean'])
            print( 'mdn:%0.2f' % v['Harmonic mean'].median(),
                  'iqr:%0.2f' % iqr(v['Harmonic mean']), 'p:%0.5f' % p, 's:%0.5f' % s,_, k, k1)
for _, __ in dfh.groupby(['DataSet']):
    print(_)
    for k1, v1 in __.groupby('Algorithms'):
        print(k1,'mdn:%0.2f' % v1['Harmonic mean'].median(),
                  'iqr:%0.2f' % iqr(v1['Harmonic mean']))