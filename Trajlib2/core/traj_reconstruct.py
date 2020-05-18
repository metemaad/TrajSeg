import pandas as pd
import numpy as np
from Trajlib2.core.utils import calculate_two_point_distance, get_bearing_points
from matplotlib import pyplot as plt


def func(start, l, t, ts):
    (lat, lon) = start

    t = np.radians(np.random.normal(t, ts, 1))
    dy = l * np.cos(t)
    dx = l * np.sin(t)
    r_earth = 6378000
    pi = np.pi
    new_latitude = lat + (dy / r_earth) * (180 / pi)
    new_longitude = lon + (dx / r_earth) * (180 / pi) / np.cos(lat * pi / 180);
    return new_latitude, new_longitude


def reconst(start=(0, 0, 0, 0, 0), end=(1, 1, 0, 0, 0), density=0.01, s=0.1,
            plot=False, time_threshold=2000):
    a = []
    b = []
    c = []
    d = []
    e = []
    x, y, ts, l, st = start
    p, q, tsp, lp, et = end
    # print(st,et)
    if et == st:
        return None
    delta = st - et
    if np.abs(delta.seconds) > time_threshold:
        res = [[x, y, ts, l, st], [p, q, tsp, lp, et]]
        return pd.DataFrame(res, columns=['lat', 'lon', 'tsid', 'label', 'time'])

    n = int(delta.seconds * density)
    if n<1:
        n=1
    ptd = pd.Timedelta(((et - st).seconds / n) * 1000000000)

    for i in range(n - 1):

        e.append(st + ptd * (i + 1))

        ll = calculate_two_point_distance(start[0], start[1], end[0], end[1]) - calculate_two_point_distance(start[0],
                                                                                                             start[1],
                                                                                                             x, y)
        br = get_bearing_points(x, y, end[0], end[1])
        x, y = func((x, y), ll / (n - i), br, br * s)

        a.append(x[0].astype(float))
        b.append(y[0].astype(float))
        if tsp == ts:
            c.append(ts)
        else:
            if i < int(n / 2):
                c.append(ts)
            else:
                c.append(tsp)
        if lp == l:
            d.append(l)
        else:
            if i < int(n / 2):
                d.append(l)
            else:
                d.append(lp)
        # print(x,y,ll,br)
    if plot:
        from matplotlib import pyplot as plt
        plt.scatter(a, b, color='b', s=5)
        plt.scatter(start[0], start[1], color='r')
        plt.scatter(end[0], end[1], color='g')
    re = list(zip(a, b, c, d, e))
    return pd.DataFrame(re, columns=['lat', 'lon', 'tsid', 'label', 'time'])


def reconstruct_traj_df(df,density=0.001, std=0.002, time_threshold=2000):

    v=df

    tsid =  v.TSid
    label = v.label
    timea = v.index

    datediff = np.diff(timea) / 1000000000
    lat = v.lat
    lon = v.lon
    dfo=pd.DataFrame()
    for x in range(len(datediff)):

        start = (lon[x], lat[x], tsid[x], label[x], timea[x])
        end = (lon[x + 1], lat[x + 1], tsid[x + 1], label[x + 1], timea[x + 1])
        if start == end:
            continue

        res2 = reconst(start, end, density=density, s=std, plot=True, time_threshold=time_threshold)
        dfo = dfo.append(res2)

        dfo = dfo.reset_index()
        dfo = dfo.drop(columns=['index'])
    # print(df)
    return dfo
def reconstruct_traj(filename="~/Trajlib2/Trajlib2/databases/fishing/fv_d3.txt",
                     sep=';', density=0.001, std=0.002, time_threshold=2000):
    df = pd.read_csv(filename, sep=sep)
    df = df.assign(dt=pd.to_datetime(df.time))
    df = df.set_index(['tid', 'dt'])
    df = df.sort_index()
    df = df.reset_index()
    # time_threshold=2000

    dfo = pd.DataFrame()
    for k,v in df.groupby(['tid']):
        v = v.assign(dt=pd.to_datetime(df.time))
        v = v.set_index(['dt'])
        v = v.sort_index()
        v = v.reset_index()
        tsid = v.tid * 100000 + v.sid
        label = v.label
        timea = pd.to_datetime(v.time)

        datediff = np.diff(pd.to_datetime(v.time)) / 1000000000
        lat = v.latitude
        lon = v.longitude

        for x in range(len(datediff)):

            start = (lon[x], lat[x], tsid[x], label[x], timea[x])
            end = (lon[x + 1], lat[x + 1], tsid[x + 1], label[x + 1], timea[x + 1])
            if start == end:
                continue

            res2 = reconst(start, end, density=density, s=std, plot=True, time_threshold=time_threshold)
            dfo = dfo.append(res2)

            dfo = dfo.reset_index()
            dfo = dfo.drop(columns=['index'])
        # print(df)
    return dfo


def plot_df(df,seg_id,tsid,df2=None,df2_segid=None,title="",path="/Users/owner/Trajlib2/Trajlib2/databases/fishing",traj_id="1"):
    df=df.assign(segid=seg_id)
    df = df.assign(sid=tsid)
    import matplotlib.pyplot as plt

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

    import matplotlib.colors as colors
    colors_list = list(colors._colors_full_map.values())

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))



    color = colors_list#['g', 'b', 'k', 'pink', 'yellow']
    i = 0
    for k, v in df.groupby(['label']):
        i = (i + 1) % len(colors_list)
        # print(i)
        axes[0].scatter(v.lat, v.lon, s=200, color=color[i], alpha=.9)
        axes[1].scatter(v.lat, v.lon, s=200, color=color[i], alpha=.9)
    colors_list=colors_list[i:]
    color = colors_list# ['orange', 'red', 'k', 'pink', 'yellow']
    i = 0
    for k, v in df.groupby(['segid']):
        i = (i + 10) % len(colors_list)
        # print(i)
        axes[0].scatter(v.lat, v.lon, s=30, color=color[i], alpha=.9)
        axes[0].set_xlabel("Discovered segments")
    i = 0
    for k, v in df.groupby(['sid']):
        i = (i + 10) % len(colors_list)
        # print(i)
        axes[1].set_xlabel("Ground truth segments")
        axes[1].scatter(v.lat, v.lon, s=30, color=color[i], alpha=.9)
    color = list(colors.BASE_COLORS.values())[::-1]
    if df2 is not None:
        df2 = df2.assign(segid=df2_segid)
        i=0
        for k, v in df2.groupby(['segid']):
            i = (i + 1) % len(colors_list)
            # print(i)
            axes[0].scatter(v.lon, v.lat, s=550, color=color[i], alpha=.8)
            #axes[0].set_title("Discovered segments")

    # plt.xlim([-35.15,-34.19])
    # plt.ylim([0,1.3])
    #axes[0].axis('off')
    #axes[1].axis('off')
    fig.tight_layout()
    #plt.title(title)
    plt.savefig(path+'/tg/'+traj_id+'.png')
    plt.show()
def convert():
    for i in range(10):
        df = reconstruct_traj(filename="~/Trajlib2/Trajlib2/databases/fishing/fv_d"+str(i+1)+".txt",
                              density=0.00005, time_threshold=88000)
        df.to_csv("~/Trajlib2/Trajlib2/databases/fishing/fv_d_recons"+str(i+1)+'.csv')
        print(i + 1, df.shape)
        plot_df(df,df.tsid,df.label)
#convert()
from Trajlib2.databases import load_datasets
for _ in load_datasets.load_data_fishing_data(path="~/Trajlib2/Trajlib2/databases/fishing/"):
    print(_.shape)
    plot_df(_,_.TSid,_.label,None,None,"title",traj_id="1test")
    break
