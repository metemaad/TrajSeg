import numpy as np
from sklearn.model_selection import ParameterGrid

from Trajlib2.SegmentationAlgorithms.SegmentationByCBSMoT.CBSMoT import CBSmot
from Trajlib2.SegmentationEvaluation import harmonic_mean


class SegmentationByCBSMoT:
    def __init__(self, max_dist=100, area=0.5, min_time=60, time_tolerance=60, merge_tolerance=100, verbose=False):
        self.max_dist = max_dist
        self.area = area
        self.min_time = min_time
        self.time_tolerance = time_tolerance
        self.merge_tolerance = merge_tolerance
        self.verbose = verbose
        self.segment_id = None

    @staticmethod
    def core(trajectory, max_dist=100, area=0.5, min_time=60, time_tolerance=60, merge_tolerance=100,
             verbose=False):
        # print("",max_dist,area,min_time,time_tolerance,merge_tolerance)
        cbsmote = CBSmot()
        eps = max_dist
        if eps is None:
            eps = CBSmot.get_quantile(trajectory, area)
        index, stops = cbsmote.segment_stops_moves(trajectory, eps, min_time, time_tolerance, merge_tolerance)
        index_move = []
        moves = []
        start = 0
        for valor in index:
            if valor[0] == trajectory.index[start]:
                start = trajectory.index.get_loc(valor[1])
                if type(start) is slice:
                    start = start.start
                start += 1
                continue
            end = trajectory.index.get_loc(valor[0])
            if type(end) is slice:
                end = end.stop
            end += 1
            moves.append(trajectory.loc[trajectory.index[start]:trajectory.index[end], :])
            index_move.append([trajectory.index[start], trajectory.index[end]])
            start = trajectory.index.get_loc(valor[1])
            if type(start) is slice:
                start = start.start
            start += 1

        positions = index + index_move
        last_idx = 0
        segments = []
        for p in positions:
            idx = trajectory.index.get_loc(p[1])
            if type(idx) is slice:
                idx = idx.stop
            segments.append([last_idx, idx])
            last_idx = idx + 1
        segments.append([last_idx, len(trajectory)])
        # print("segments:",segments)
        segment_id = np.zeros(segments[len(segments) - 1][1])
        segment_label = 0
        for s in segments:
            segment_id[s[0]:s[1] + 1] = segment_label
            segment_label += 1
        # print("id:",segment_id)
        return segment_id

    def tuning(self, tuning_trajectory, tuning_ground_truth=None, cbsmot_params=None, verbose=False):
        if cbsmot_params is None:
            cbsmot_params = {'area_param': [0.5],
                             'min_time_param': [60],
                             'time_tolerance_param': [60],
                             'merge_tolerance_param': None,
                             'max_dist_param': None}
        if verbose:
            print("Tuning..")

        assert (len(tuning_ground_truth[0]) == tuning_trajectory.shape[
            0]), "pass the correct labels: length of passed labels is  " + str \
            (len(tuning_ground_truth[0])) + " and length of X is  " + str(tuning_trajectory.shape[0])

        results = {}
        best = None
        best_param = None

        parm_grid = ParameterGrid(cbsmot_params)
        for p in parm_grid:

            params = {'max_dist': p['max_dist_param'], 'area': p['area_param'], 'min_time': p['min_time_param'],
                      'time_tolerance': p['time_tolerance_param'], 'merge_tolerance': p[
                    'merge_tolerance_param']}
            test_param = (p['max_dist_param'], p['area_param'], p['min_time_param'], p['time_tolerance_param'],
                          p['merge_tolerance_param'])
            results[test_param] = self.score(trajectory=tuning_trajectory, ground_truth=tuning_ground_truth, **params,
                                             verbose=verbose)
            print(results[test_param], ":", test_param)
            if best is None:
                best = results[test_param]
                best_param = test_param

            else:
                if best < results[test_param]:
                    best = results[test_param]
                    best_param = test_param

        self.max_dist = best_param[0]
        self.area = best_param[1]
        self.min_time = best_param[2]
        self.time_tolerance = best_param[3]
        self.merge_tolerance = best_param[4]
        if verbose:
            print("best:", best_param, " with ", best)
        self.segment_id = self.core(tuning_trajectory, max_dist=self.max_dist, area=self.area, min_time=self.min_time,
                                    time_tolerance=self.time_tolerance, merge_tolerance=self.merge_tolerance)

        return self

    def predict(self, trajectory, verbose=False):
        segment_id = self.core(trajectory, max_dist=self.max_dist, area=self.area, min_time=self.min_time,
                               time_tolerance=self.time_tolerance, merge_tolerance=self.merge_tolerance,
                               verbose=verbose)
        return segment_id

    def score(self, trajectory, ground_truth, verbose=False, **kwargs):
        segment_id = self.core(trajectory, **kwargs)
        h = harmonic_mean(segments=segment_id, tsid=ground_truth[0], label=ground_truth[1])
        if verbose:
            print("Scoring " + ','.join('{0}={1!r}'.format(k, kwargs[k]) for k in kwargs), str(h))
        return h
