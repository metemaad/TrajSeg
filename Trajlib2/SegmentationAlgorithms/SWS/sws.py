import numpy as np

from Trajlib2.SegmentationAlgorithms.SWS.kernels import random_walk, kinematic, linear, cubic, linear_regression
from Trajlib2.SegmentationEvaluation import harmonic_mean


def plot_seven(lat, lon, p1, p2, pc):
    pass


def ows_adjustment(segment_id):
    i = 0
    adjusted_segment_id = segment_id
    while i < len(segment_id):
        j = i
        k = 0
        while segment_id[i] == -1:
            k = k + 1
            i = i + 1
            if len(segment_id) <= i:
                break

        if k > 0 and k < 4:

            if len(segment_id[j:k + j]) == 1:
                adjusted_segment_id[j] = segment_id[j - 1]
            else:
                if k % 2 == 0:
                    if j + k + 1 > len(segment_id):
                        next_segment_id = segment_id[j - 1]
                    else:
                        next_segment_id = segment_id[j + k + 1]
                    adjusted_segment_id[j:j + int(k / 2)] = [segment_id[j - 1]] * (int(k / 2))
                    adjusted_segment_id[j + int(k / 2):j + k] = [next_segment_id] * (int(k / 2))
                else:
                    if j + k + 1 > len(segment_id):
                        next_segment_id = segment_id[j - 1]
                    else:
                        next_segment_id = segment_id[j + k + 1]
                    adjusted_segment_id[j:j + int(k / 2) + 1] = [segment_id[j - 1]] * (int(k / 2) + 1)
                    adjusted_segment_id[j + int(k / 2) + 1:j + k] = [next_segment_id] * (int(k / 2))
        i = i + 1
        if len(segment_id) <= i:
            break

    return adjusted_segment_id


class SWS:
    interpolations = [linear_regression.linear_regression, random_walk.random_walk2, cubic.cubic, linear.linear,
                      kinematic.kinematic]
    interpolation_names = ['LR', 'Random Walk', 'cubic', 'linear', 'kinematic']

    def generate_error_signal(self, df, interpolation_name='Random Walk', window_size=7, verbose=False):

        f = self.interpolation_names.index(interpolation_name)
        res = {}
        means_res = {}

        da = self.calculate_error(df, f1=self.interpolations[f], rang=(0, 0), plot=False, window_size=window_size,
                                  verbose=verbose)
        res[self.interpolation_names[f]] = da
        means_res[self.interpolation_names[f]] = np.nanmax(da)

        # use the first and last calculated error instead of zero
        da[0:int(window_size / 2)] = [da[int(window_size / 2)]] * int(window_size / 2)
        da[len(da) - 1 * int(window_size / 2) - 1:len(da)] = [da[len(da) - int(window_size / 2) - 2]] * (
                int(window_size / 2) + 1)

        return da

    @staticmethod
    def calculate_error(df, f1=None, rang=(0, 0), plot=False, window_size=7, verbose=False):
        start = rang[0]
        if rang[1] == 0:
            limit = df.shape[0]
        else:
            limit = rang[1]
        end = limit
        ln = int(window_size / 2)
        da = [0] * ln

        for ix in range(end - start - window_size):
            try:
                seven_points = df.iloc[start + ix:start + ix + window_size, :]
                lat = seven_points.lat.values
                lon = seven_points.lon.values
                p1, p2, pc, d = f1(seven_points, verbose=verbose)
                if plot:
                    plot_seven(lat, lon, p1, p2, pc)
            except Exception as e:
                # if verbose:
                print("Error:", e)
                d = 0
            da.append(d)

        for i in range(ln + 1):
            da.append(0)
        return da

    @staticmethod
    def find_sub_segments(curr, ep):

        curr = np.array(curr)
        p = []
        q = []
        old_up_down = (ep <= curr[0])
        j = 0
        k = 0
        while j < len(curr):
            up_down = (ep <= curr[j])
            if old_up_down != up_down:
                if (curr[k:j] <= ep).all():
                    if j - k == 1:
                        q.append((k, k))
                    else:
                        p.append((k, j - 1))
                else:
                    if (curr[k:j] > ep).all():
                        if j - k == 1:
                            q.append((k, k))
                        else:
                            p.append((k, j - 1))
                k = j
            old_up_down = up_down
            j = j + 1
        if (curr[k:j] <= ep).all():
            if j - k == 1:
                q.append((k, k))
            else:
                p.append((k, j - 1))
        else:
            if (curr[k:j] > ep).all():
                if j - k == 1:
                    q.append((k, k))
                else:
                    p.append((k, j))
        return p, q

    def sws_segmentation(self, dataframe, error_array, rang=(0, 0), ep=20000, verbose=False):
        start = rang[0]
        if rang[1] == 0:
            limit = dataframe.shape[0]
        else:
            limit = rang[1]

        q = [(start, limit)]
        p = []

        while len(q) != 0:
            t = q.pop()
            curr = error_array[t[0]:t[1]]
            if len(curr) == 0:
                continue
            m = max(curr)
            if m >= ep:
                ix = list(np.where(np.array(curr) == m)[0])
                ix.sort()
                if len(ix) <= 1:

                    q.append((t[0], t[0] + ix[0]))
                    q.append((t[0] + ix[0] + 1, t[1]))
                else:
                    mp, mq = self.find_sub_segments(curr, ep)
                    for _ in mp:
                        p.append(_)
                    for _ in mq:
                        q.append(_)
            else:
                p.append(t)

        segment_id = self.generate_cluster_labels(p, dataframe.shape[0])

        return segment_id

    @staticmethod
    def generate_cluster_labels(trajectory_set, n):
        i = 1
        labels = [-1] * n
        for ts in trajectory_set:
            if ts[1] == n:
                labels[ts[0]:] = [i] * (n - ts[0])
            else:
                labels[ts[0]:ts[1] + 1] = [i] * (ts[1] - ts[0] + 1)
            i = i + 1
        # if len(np.where(np.array(labels)==-1)[0]):
        #    print("number of noises:",len(np.where(labels==-1)[0]))
        # labels=ows_adjustment(labels)
        return labels

    def __init__(self):
        # load data
        pass
        # print("running ows....")

    @staticmethod
    def get_eps_candidates(error_signal):
        e = [np.nanpercentile(np.abs(error_signal), 99.99),np.nanpercentile(np.abs(error_signal), 99.9),np.nanpercentile(np.abs(error_signal), 99.5),
            np.nanpercentile(np.abs(error_signal), 99), np.nanpercentile(np.abs(error_signal), 98),
            np.nanpercentile(np.abs(error_signal), 98.5), np.nanpercentile(np.abs(error_signal), 97.5),
             np.nanpercentile(np.abs(error_signal), 97), np.nanpercentile(np.abs(error_signal), 96),
             np.nanpercentile(np.abs(error_signal), 95), np.nanpercentile(np.abs(error_signal), 94),
             np.nanpercentile(np.abs(error_signal), 93), np.nanpercentile(np.abs(error_signal), 92),
             np.nanpercentile(np.abs(error_signal), 91), np.nanpercentile(np.abs(error_signal), 90)
             ]
        ix=[99.99,99.9,99.5,99,98,98.5,97.5,97,96,95,94,93,92,91,90]
        return list(zip(e,ix))

    def get_best_eps(self, error_signal, train_data, ground_truth, verbose=False):

        ep_a = self.get_eps_candidates(error_signal)

        epsa = {}
        epsp = {}
        for _,ix_ in ep_a:
            segmet_id = self.sws_segmentation(dataframe=train_data, error_array=error_signal,
                                              ep=_, verbose=verbose)
            h = harmonic_mean(segments=segmet_id, tsid=ground_truth[0], label=ground_truth[1])

            epsa[_] = h
            epsp[_] = ix_

        epsa_a = None
        epsa_k = None
        epsa_p = None
        for k in epsa.keys():
            if epsa_a is None:
                epsa_a = epsa[k]
                epsa_k = k
                epsa_p=epsp[k]
            else:
                if epsa_a <= epsa[k]:
                    epsa_a = epsa[k]
                    epsa_k = k
                    epsa_p = epsp[k]
        if verbose:
            print('best eps:', epsa_a, ' for candidate : ', epsa_k,"p:",epsa_p)
        return epsa_a, epsa_k,epsa_p
