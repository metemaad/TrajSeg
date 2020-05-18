from sklearn.model_selection import ParameterGrid
from Trajlib2.SegmentationAlgorithms.WarpedKMeans.wkmb import WKM
from Trajlib2.SegmentationAlgorithms.WarpedKMeans.mathlib import whiten_df
from Trajlib2.SegmentationEvaluation import harmonic_mean



def boundaries_to_segment_id(boundaries, length):
    seg_id = 1
    segment_ids = []
    for i in range(len(boundaries)):
        if i == len(boundaries) - 1:
            end = length
        else:
            end = boundaries[i + 1]
        s = [seg_id] * (end - boundaries[i])
        segment_ids = segment_ids + s
        seg_id = seg_id + 1

    return segment_ids


class SegmentationByWKMeans:
    def __init__(self, num_k=1, delta=0, columns=['lat','lon']):
        self.num_k = num_k  # number of clusters
        self.delta = delta
        self.columns = columns
        self.segment_id=None

    @staticmethod
    def core(trajectory, num_k, delta=0, columns=['lat', 'lon'], verbose=False):
        w = WKM(whiten_df(trajectory, columns=columns), num_k, delta)
        w.cluster()
        boundaries = w.boundaries
        segments2 = boundaries_to_segment_id(boundaries, trajectory.shape[0])
        return segments2

    def tuning(self, tuning_trajectory, tuning_ground_truth=None, wkmeans_params=None, verbose=None):
        if wkmeans_params is None:
            wkmeans_params = {'num_k_param':[len(set(tuning_ground_truth))],'delta_param':[0]}
        if verbose:
            print("Tuning..")


        assert (len(tuning_ground_truth[0]) == tuning_trajectory.shape[0]), "pass the correct labels: length of passed labels is " + str(
            len(tuning_ground_truth[0])) + " and length od X is " + str(tuning_trajectory.shape[0])

        results = {}
        best = None
        best_param = None

        parm_grid = ParameterGrid(wkmeans_params)
        for p in parm_grid:
            if p['num_k_param'] is None:
                params = {'num_k': len(set(tuning_ground_truth[0])), 'delta': p['delta_param']}
                best_param = (len(set(tuning_ground_truth[0])), p['delta_param'])
            else:
                params = {'num_k': p['num_k_param'], 'delta': p['delta_param']}
                test_param = (p['num_k_param'], p['delta_param'])
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

        self.num_k = best_param[0]*9
        self.delta = best_param[1]
        print("best:", best_param, " with ", best)


        return self

    def predict(self, trajectory, verbose=False):
        segment_id = self.core(trajectory, num_k=self.num_k,delta=self.delta,columns=self.columns,verbose=verbose)
        return segment_id

    def score(self, trajectory,ground_truth, verbose=False, **kwargs):
        segment_id = self.core(trajectory, **kwargs)
        h = harmonic_mean(segments=segment_id, tsid=ground_truth[0], label=ground_truth[1])
        if verbose:
            print("Scoring " + ','.join('{0}={1!r}'.format(k, kwargs[k]) for k in kwargs), str(h))
        return h
