from sklearn.model_selection import ParameterGrid

from Trajlib2.SegmentationAlgorithms.GRASP_UTS.GRASPUTS import GRASPUTS, GRASPUTSEvaluate
from Trajlib2.SegmentationEvaluation import harmonic_mean


class SegmentationByGRASPUTS:
    def __init__(self, best_param=None, lat_col='lat', lon_col='lon', time_format="%Y-%m-%d %H:%M:%S",
                 feature_names=['direction_inference', 'speed_inference_m_s'],
                 alpha=0.3, max_iterations=30, min_time=6, partitioning_factor=0, join_coefficient=0.3, verbose=False):
        self.verbose = verbose

        if best_param is not None:
            self.alpha = best_param[0]
            self.max_iterations = best_param[2]
            self.min_time = best_param[3]
            self.partitioning_factor = best_param[4]
            self.join_coefficient = best_param[1]
        else:
            self.alpha = alpha
            self.max_iterations = max_iterations
            self.min_time = min_time
            self.partitioning_factor = partitioning_factor
            self.join_coefficient = join_coefficient

        if lat_col is None:
            lat_col = 'lat'
        if lon_col is None:
            lon_col = 'lon'
        if feature_names is None:
            feature_names = ['direction_inference', 'speed_inference_m_s']

        self.lat_col = lat_col
        self.lon_col = lon_col
        self.time_format = time_format
        self.feature_names = feature_names
        self.segment_id = None

    @staticmethod
    def core(trajectory, feature_names, lat_col='lat', lon_col='lon',
             time_format='%Y-%m-%d %H:%M:%S',
             min_time=6,
             join_coefficient=0.3,
             alpha=0.3,
             max_iter=30,
             partitioning_factor=0, verbose=False):

        grasputs = GRASPUTS(feature_names=feature_names,
                            lat_col=lat_col, lon_col=lon_col,
                            time_format=time_format,
                            min_time=min_time,
                            join_coefficient=join_coefficient,
                            alpha=alpha,
                            partitioning_factor=partitioning_factor, max_iter=max_iter)

        segments, cost = grasputs.segment(trajectory)

        segment_id = GRASPUTSEvaluate.get_predicted(segments)

        return segment_id

    def tuning(self, tuning_trajectory, tuning_ground_truth=None, grasputs_params=None, verbose=False):

        if grasputs_params is None:
            grasputs_params = {'alpha': [self.alpha],
                               'partitioning_factor': [self.partitioning_factor],
                               'max_iterations': [self.max_iterations],
                               'min_time': [self.min_time],
                               'jcs': [self.join_coefficient]}
        if verbose:
            print("Tuning..")

        assert (len(tuning_ground_truth[0]) == tuning_trajectory.shape[
            0]), "pass the correct labels: length of passed labels is " + str(
            len(tuning_ground_truth[0])) + " and length od X is " + str(tuning_trajectory.shape[0])

        results = {}
        best = None
        best_param = None

        parm_grid = ParameterGrid(grasputs_params)
        for p in parm_grid:
            grasputs = {'feature_names': self.feature_names,
                        'lat_col': self.lat_col,
                        'lon_col': self.lon_col,
                        'time_format': self.time_format,
                        'min_time': p['min_time'],
                        'max_iter': p['max_iterations'],
                        'join_coefficient': p['jcs'],
                        'alpha': p['alpha'],
                        'partitioning_factor': p['partitioning_factor']}
            test_param = (p['min_time'], p['max_iterations'], p['jcs'], p['alpha'], p['partitioning_factor'])
            results[test_param] = self.score(trajectory=tuning_trajectory, ground_truth=tuning_ground_truth,
                                             verbose=verbose, **grasputs)

            print(results[test_param], ":", test_param)
            if best is None:
                best = results[test_param]
                best_param = test_param

            else:
                if best < results[test_param]:
                    best = results[test_param]
                    best_param = test_param

        self.alpha = best_param[3]
        self.max_iterations = best_param[1]
        self.min_time = best_param[0]
        self.partitioning_factor = best_param[4]
        self.join_coefficient = best_param[2]

        print("best:", best_param, " with ", best)

        return self

    def predict(self, trajectory, verbose=False):

        segment_id = self.core(trajectory=trajectory, alpha=self.alpha,
                               max_iter=self.max_iterations,
                               min_time=self.min_time,
                               partitioning_factor=self.partitioning_factor, join_coefficient=self.join_coefficient,
                               feature_names=self.feature_names, lon_col=self.lon_col, lat_col=self.lat_col,
                               time_format=self.time_format, verbose=verbose)
        return segment_id

    def score(self, trajectory, ground_truth, verbose=False, **kwargs):
        segment_id = self.core(trajectory, **kwargs)
        h = harmonic_mean(segments=segment_id, tsid=ground_truth[0], label=ground_truth[1])
        if verbose:
            print("Scoring " + ','.join('{0}={1!r}'.format(k, kwargs[k]) for k in kwargs), str(h))
        return h
