import numpy as np
from Trajlib2.SegmentationAlgorithms.SWS.sws import SWS
from Trajlib2.SegmentationEvaluation import harmonic_mean
from sklearn.model_selection import ParameterGrid


class SegmentationBySWS:
    def __init__(self, epsilon=None, window_size=7, interpolation_kernel='linear', verbose=False,percentile=None):
        self.interpolation_kernel = interpolation_kernel
        self.epsilon = epsilon
        self.window_size = window_size
        self.verbose = verbose
        self.segment_id = None
        self.percentile = percentile

    def get_eps_candidates(self, error_signal):
        if self.verbose:
            print("Epsilon candidates set to 90,..99 percentiles.")
        e = [np.nanpercentile(np.abs(error_signal), 99), np.nanpercentile(np.abs(error_signal), 98),
             np.nanpercentile(np.abs(error_signal), 97), np.nanpercentile(np.abs(error_signal), 96),
             np.nanpercentile(np.abs(error_signal), 95), np.nanpercentile(np.abs(error_signal), 92),
             np.nanpercentile(np.abs(error_signal), 90)]
        return e

    @staticmethod
    def core(trajectory, interpolation_kernel='linear', window_size=7,
             epsilon=None, verbose=False,percentile=None):

        s = SWS()

        if verbose:
            print("generate error signal... ws:", str(window_size), " int:", interpolation_kernel)

        error_signal = s.generate_error_signal(trajectory, interpolation_name=interpolation_kernel,
                                               window_size=window_size)

        if percentile is not None:
            epsilon=np.nanpercentile(np.abs(error_signal), percentile)
        if verbose:
            print("da_train", len(error_signal))
            print("da_train", error_signal)

            print("Segmentation...  ")
        seg_id = s.sws_segmentation(dataframe=trajectory, error_array=error_signal, ep=epsilon)
        return seg_id

    def tuning(self, tuning_trajectory, tuning_ground_truth=None, sws_params=None,
               verbose=False):
        if sws_params is None:
            sws_params = {'window_size_param': [7],
                          'interpolation_kernel_param': ['linear'],
                          'epsilon_param': [None]}

        if verbose:
            print("Tuning..")

        assert (len(tuning_ground_truth[0]) == tuning_trajectory.shape[
            0]), "pass the correct labels: length of passed labels is " + str(
            len(tuning_ground_truth[0])) + " and length od X is " + str(tuning_trajectory.shape[0])

        results = {}
        best = None
        best_param = None



        parm_grid = ParameterGrid(sws_params)
        for p in parm_grid:

            s=SWS()
            error_signal = s.generate_error_signal(tuning_trajectory, interpolation_name=p['interpolation_kernel_param'],
                                                       window_size=p['window_size_param'])
            if p['epsilon_param'] is None:
                epsilon = s.get_eps_candidates(error_signal)
                for ep,ix in epsilon:
                    params = {'window_size': p['window_size_param'],
                              'interpolation_kernel': p['interpolation_kernel_param'],
                              'epsilon':ep}
                    seg_id = s.sws_segmentation(dataframe=tuning_trajectory,
                                                error_array=error_signal,
                                                ep=ep, verbose=verbose)
                    test_param = (p['window_size_param'], p['interpolation_kernel_param'], ep,ix)
                    results[test_param] = harmonic_mean(segments=seg_id, tsid=tuning_ground_truth[0], label=tuning_ground_truth[1])

                    print(results[test_param],":",test_param)
                    if best is None:
                        best = results[test_param]
                        best_param = test_param

                    else:
                        if best < results[test_param]:
                            best = results[test_param]
                            best_param=test_param
            else:
                params = {'window_size': p['window_size_param'],
                          'interpolation_kernel': p['interpolation_kernel_param'],
                          'epsilon': p['epsilon_param'],'percentile':None}
                test_param = (p['window_size_param'], p['interpolation_kernel_param'], p['epsilon_param'],None)
                results[test_param] = self.score(trajectory=tuning_trajectory, ground_truth=tuning_ground_truth,
                                                 **params,
                                                 verbose=verbose)
                print(results[test_param],":",test_param)
                if best is None:
                    best = results[test_param]
                    best_param=test_param

                else:
                    if best < results[test_param]:
                        best = results[test_param]
                        best_param=test_param


        self.window_size = best_param[0]
        self.interpolation_kernel = best_param[1]
        self.epsilon = best_param[2]
        self.percentile=best_param[3]

        if verbose:
            print("best:", best_param, " with ", best)

        return self

    def predict(self, trajectory,percentile, verbose=False):
        if percentile is not None:
            self.percentile=percentile
        segment_id = self.core(trajectory, window_size=self.window_size,interpolation_kernel=self.interpolation_kernel,epsilon=self.epsilon, verbose=verbose,percentile=percentile)
        return segment_id

    def score(self, trajectory, ground_truth, verbose=False, **kwargs):
        segment_id = self.core(trajectory, **kwargs)
        h = harmonic_mean(segments=segment_id, tsid=ground_truth[0], label=ground_truth[1])
        if verbose:
            print("Scoring " + ','.join('{0}={1!r}'.format(k, kwargs[k]) for k in kwargs), str(h))
        return h


"""            
        for interpolation_method in interpolation_kernel_param:
            for ws in window_size_param:
                # generate error signal
                s = SWS()
                if verbose:
                    print("generate error signal... ws:", str(ws), " int:", interpolation_method)
                error_signal = s.generate_error_signal(tuning_trajectory, interpolation_name=interpolation_method,
                                                       window_size=ws, verbose=verbose)
                if (epsilon_param is None) or (epsilon_param == [None]):
                    epsilon_param = self.get_eps_candidates(error_signal)
                if verbose:
                    print("epsilon candidates: ", epsilon_param)
                for eps in epsilon_param:

                    seg_id = s.sws_segmentation(dataframe=tuning_trajectory,
                                                error_array=error_signal,
                                                ep=eps, verbose=verbose)
                    h = harmonic_mean(segments=seg_id, tsid=tuning_ground_truth[0], label=tuning_ground_truth[1])
                    results[(interpolation_method, ws, eps)] = h
                    if verbose:
                        print(interpolation_method, ws, eps, h)

                    if best is None:
                        best = results[(interpolation_method, ws, eps)]
                        best_param = (interpolation_method, ws, eps)
                    else:
                        if best < results[(interpolation_method, ws, eps)]:
                            best = results[(interpolation_method, ws, eps)]
                            best_param = (interpolation_method, ws, eps)

        self.interpolation_kernel = best_param[0]
        self.window_size = best_param[1]
        self.epsilon = best_param[2]
"""
