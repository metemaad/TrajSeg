from pandas import to_datetime
import numpy as np
from numpy import where
from Trajlib2.SegmentationEvaluation import harmonic_mean




class SegmentationByTime():
    def __init__(self, mean_time=10, minimum_number_of_items_in_each_traj=10):
        self.mean_time = mean_time
        self.minimum_number_of_items_in_each_traj = minimum_number_of_items_in_each_traj

    def _segment_by_time(self, trajectory, time_field='collected_time', mean_time=None,
                         minimum_number_of_items_in_each_traj=10):
        t1 = to_datetime(trajectory.loc[:, ['time_date']].values.ravel())
        t2 = t1[1:]
        t1 = t1[:-1]
        sec = (t2 - t1).seconds
        if mean_time is None:
            mean_time = np.mean(sec)
            print("mean_time set to ", str(mean_time))

        g = where(sec > mean_time)
        start_segment = 0
        segments = []
        for i in range(len(g[0])):
            if g[0][i] - start_segment >= minimum_number_of_items_in_each_traj:
                segments.append([start_segment, g[0][i]])
                start_segment = g[0][i] + 1
        segments.append([start_segment, trajectory.shape[0]])
        trajectory_segments = {}
        i = 0
        for segment in segments:
            trajectory_segments[i] = trajectory.iloc[segment[0]:segment[1], :]
            i = i + 1
        del trajectory
        del t1
        del t2
        del sec

        labels = np.zeros(segments[len(segments) - 1][1])
        l = 0
        for s in segments:
            labels[s[0]:s[1] + 1] = l
            l += 1

        return labels

    def tuning(self, X, y=None, params=None):
        print("tuning..")

        ground_truth = y
        results = {}
        best = None
        best_min_time = None
        for param in params:
            results[param] = self.score(X, ground_truth, mean_time=param)
            if best is None:
                best = results[param]
                best_min_time = param
            else:
                if best < results[param]:
                    best = results[param]
                    best_min_time = param

        self.mean_time = best_min_time
        print("best:",best_min_time," with ",best)
        self.segment_id = self._segment_by_time(X, mean_time=self.mean_time,
                                                minimum_number_of_items_in_each_traj=self.minimum_number_of_items_in_each_traj)

        return self

    def predict(self, X, y=None):
        segment_id = self._segment_by_time(X, mean_time=self.mean_time,
                                           minimum_number_of_items_in_each_traj=self.minimum_number_of_items_in_each_traj)
        return segment_id

    def score(self, X, ground_truth, **kwargs):


        segment_id = self._segment_by_time(X, **kwargs)
        h = harmonic_mean(ground_truth, segment_id)
        #print("Scoring "+','.join('{0}={1!r}'.format(k, kwargs[k]) for k in kwargs),str(h))
        return h
