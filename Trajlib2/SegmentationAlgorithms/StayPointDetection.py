# Stay Point Detection Algorithm

# Traj is a GPS Trajectory assuming is a pandas dataframe
# Theta_d is a distance threshold   (metric meter)
# Theta_t is a time span threshold  (metric seconds)
# output S a set of stay points

import numpy as np
from sklearn.model_selection import ParameterGrid
from Trajlib2.SegmentationEvaluation import harmonic_mean
from Trajlib2.core.utils import haversine, StayPoint


class SegmentationByStayPointDetection:
    def __init__(self, theta_distance=200, theta_time=600):
        self.theta_distance = theta_distance
        self.theta_time = theta_time
        self.segment_id = None

    @staticmethod
    def core(trajectory, theta_distance, theta_time, verbose=False):
        i = 0
        point_num = trajectory.shape[0]  # the number of GPS points in traj
        S = []  # set of stay points
        segments = [-1] * point_num
        stay_point_id = 0
        while i < point_num:
            j = i + 1
            while j < point_num:
                p_i = trajectory.iloc[i, :]
                p_j = trajectory.iloc[j, :]
                distance = haversine((p_i.lat, p_i.lon),
                                     (p_j.lat, p_j.lon))  # calculate haversine distance between two points
                if distance > theta_distance:
                    p_i_T = p_i.name
                    p_j_T = p_j.name
                    delta_t = (p_j_T - p_i_T).total_seconds()
                    if delta_t > theta_time:
                        s = StayPoint()
                        stay_point_id = stay_point_id + 1
                        s.lat = np.mean(trajectory.iloc[i:j, :].lat)
                        s.lon = np.mean(trajectory.iloc[i:j, :].lon)
                        s.arrive_time = p_i_T
                        s.leave_time = p_j_T
                        s.i = i
                        s.j = j
                        S.append(s)
                        for _ in range(i, j):
                            segments[_] = stay_point_id

                    break
                j = j + 1
            i = j
        segments2 = segments
        ss = np.max(segments) + 1
        i = 0
        while i < point_num:
            j = i + 1
            while (j < point_num) and (segments[j] == -1) and (segments[j] < segments[i]):
                segments2[j] = ss
                j = j + 1
            ss = np.max(segments2) + 1
            i = i + 1
        segment_id = np.add(segments2, [2] * len(segments2))
        return segment_id

    def tuning(self, tuning_trajectory, tuning_ground_truth=None, spd_params=None, verbose=False):
        if verbose:
            print("tuning..")
        if spd_params is None:
            spd_params = {'theta_distance_param': [100, 200, 500, 1000, 2000], 'theta_time_param': [60, 300, 600, 1200]}

        assert (len(tuning_ground_truth[0]) == tuning_trajectory.shape[
            0]), "pass the correct labels: length of passed labels is " + str(
            len(tuning_ground_truth[0])) + " and length od X is " + str(tuning_trajectory.shape[0])

        results = {}
        best = None
        best_param = None

        parm_grid = ParameterGrid(spd_params)
        for p in parm_grid:
            params = {'theta_distance': p['theta_distance_param'], 'theta_time': p['theta_time_param']}
            test_param = (p['theta_distance_param'], p['theta_time_param'])
            results[test_param] = self.score(trajectory=tuning_trajectory, ground_truth=tuning_ground_truth, **params,
                                             verbose=verbose)
            #if verbose:
            print(results[test_param], ":", test_param)
            if best is None:
                best = results[test_param]
                best_param = test_param
            else:
                if best < results[test_param]:
                    best = results[test_param]
                    best_param=test_param

        self.theta_distance = best_param[0]
        self.theta_time = best_param[1]

        print("best:", best_param, " with ", results[best_param])
        self.segment_id = self.core(tuning_trajectory, theta_distance=self.theta_distance, theta_time=self.theta_time)

        return self

    def predict(self, trajectory,verbose=False):
        segment_id = self.core(trajectory, theta_distance=self.theta_distance, theta_time=self.theta_time, verbose=verbose)
        return segment_id

    def score(self, trajectory, ground_truth, verbose=False, **kwargs):
        segment_id = self.core(trajectory, **kwargs)
        h = harmonic_mean(segments=segment_id, tsid=ground_truth[0], label=ground_truth[1])
        if verbose:
            print("Scoring " + ','.join('{0}={1!r}'.format(k, kwargs[k]) for k in kwargs), str(h))
        return h
