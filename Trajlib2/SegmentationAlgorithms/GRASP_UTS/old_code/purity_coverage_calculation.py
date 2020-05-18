import numpy as np

def purity(dataframe, trajectories_set, trajectory_label='transportation_mode', rang=(0, 0)):
    start = rang[0]
    if rang[1] == 0:
        limit = dataframe.shape[0]
    else:
        limit = rang[1]
    end = limit

    dataframe['flag'] = -1
    avg = []
    ci = np.where(dataframe.columns == 'flag')[0]
    i = 1
    for ts in trajectories_set:
        dataframe.iloc[ts[0]:ts[1], ci] = i
        ma = 0
        for tp in set(dataframe.iloc[ts[0]:ts[1], :][trajectory_label]):
            tmp = dataframe.iloc[ts[0]:ts[1], :]
            a = tmp.loc[tmp[trajectory_label] == tp].shape[0]
            if (a > ma):
                ma = a
        avg.append(ma * 1.0 / dataframe.iloc[ts[0]:ts[1], :].shape[0])
        i = i + 1
    return avg, np.mean(np.array(avg))


def coverage(df, actual_label='lid', rang=(0, 0)):
    start = rang[0]
    if rang[1] == 0:
        limit = df.shape[0]
    else:
        limit = rang[1]
    end = limit
    cov = []
    for ts in set(df.iloc[0:limit, :][actual_label]):
        tmp = df.loc[df[actual_label] == ts, :]
        mx = 0
        for l in set(tmp.flag):
            tmp1 = tmp.loc[tmp.flag == l, :]
            if (mx <= tmp1.shape[0]):
                mx = tmp1.shape[0]
        cov.append(mx * 1.0 / tmp.shape[0])
    return cov, np.mean(np.array(cov))