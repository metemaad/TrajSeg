from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from Trajlib2.SegmentationAlgorithms.SWS.kernels import random_walk, kinematic, linear, cubic,linear_regression
from Trajlib2.SegmentationEvaluation import harmonic_mean

interpolations = [random_walk.random_walk2, cubic.cubic, linear.linear, kinematic.kinematic,linear_regression.linear_regression]
interpolation_names = ['Random Walk', 'cubic', 'linear', 'kinematic','Linear Regression']


def check_quality_of_trajectory(tsid, label, window_size=7, verbose=False):
    bad = 0
    x = tsid[0]
    start = 0
    j = 0
    for i in range(len(tsid)):
        if x == tsid[i]:
            pass
        else:
            if verbose:
                print("[", start, ",", i, "]", i - start, label[start])
            if i - start <= window_size:
                if verbose:
                    print("***")
                bad = bad + i - start
                j = j + 1
            x = tsid[i]
            start = i + 1
    if verbose:
        print("[", start, ",", len(tsid), "]", len(tsid) - start, label[start])
    print("***:", j)
    print("quality:", int((len(tsid) - bad) / len(tsid) * 100))


def majority_vote(res, window_size=7, degree=0.9):
    start = 0
    segid = 1
    nsegid = -1
    segmentid = []
    for i in range(int(window_size / 2) + 1, len(res) - int(window_size / 2)):
        if np.sum(res[i - int(window_size / 2):i + int(window_size / 2) + 1]) >= int(window_size * degree):

            if i - start <= window_size:

                segmentid = segmentid + [nsegid] * (i - start)
                nsegid = nsegid - 1
                start = i

            else:
                segmentid = segmentid + [segid] * (i - start + 1)
                segid = segid + 1
                start = i + 1

    segmentid = segmentid + [segid] * (len(res) - start)
    return np.array(segmentid)


def calculate_error(df, f1=None, rang=(0, 0), window_size=7):
    start = rang[0]
    if rang[1] == 0:
        limit = df.shape[0]
    else:
        limit = rang[1]
    end = limit
    ln = int(window_size / 2)
    da = [0] * ln
    # print(window_size, "dd", da)
    for ix in range(end - start - window_size):
        try:
            seven_points = df.iloc[start + ix:start + ix + window_size, :]
            lat = seven_points.lat.values
            lon = seven_points.lon.values
            p1, p2, pc, d = f1(seven_points)
        except Exception as e:
            print("Error:", e)
            d = 0
        da.append(d)

    for i in range(ln + 1):
        da.append(0)
    # print(window_size, "dd", da[-10:])
    return da


def generate_error_signal(df, window_size, interpolation_name='Random Walk'):
    f = interpolation_names.index(interpolation_name)
    res = {}
    means_res = {}

    da = calculate_error(df, f1=interpolations[f], rang=(0, 0), window_size=window_size)
    res[interpolation_names[f]] = da
    means_res[interpolation_names[f]] = np.nanmax(da)
    da[0:int(window_size / 2)] = [da[int(window_size / 2)]] * int(window_size / 2)
    da[len(da) - 1 * int(window_size / 2) - 1:len(da)] = [da[len(da) - int(window_size / 2) - 2]] * (
            int(window_size / 2) + 1)

    return da


def blind_signal_to_sample(e_signal, window_size=7):
    X_train = []

    for i in range(0, len(e_signal) - window_size):
        X_train.append(e_signal[i:i + window_size])

    X_train = pd.DataFrame(X_train)
    return X_train


def signal_to_sample(e_signal, tsid, window_size=7):
    X_train = []
    y_train = []
    for i in range(0, len(e_signal) - window_size):
        X_train.append(e_signal[i:i + window_size])
        y_train.append([np.sign(len(set(tsid[i:i + window_size])) - 1)])
    y_train = pd.DataFrame(y_train)
    X_train = pd.DataFrame(X_train)
    return X_train, np.array(y_train).ravel()


def clean_short_trajectories(df_test, window_size):
    # remove all segments shorter than windows_size+1
    dic = {}
    list_of_short_trajectories = []
    for i in set(df_test.TSid):
        if df_test.iloc[np.where(np.array(df_test.TSid) == i)[0], :].shape[0] <= window_size + 1:
            dic[i] = df_test.iloc[np.where(np.array(df_test.TSid) == i)[0], :].shape[0]
            list_of_short_trajectories.append(i)
    df_test = df_test[~df_test.TSid.isin(list_of_short_trajectories)].copy()
    return df_test


class SegmentationByWSII:
    def __init__(self, window_size=7, majority_vote_degree=0.9, binary_classifier=None,kernel='Random Walk'):
        self.window_size = window_size
        self.majority_vote_degree = majority_vote_degree
        if binary_classifier is None:
            binary_classifier = RandomForestClassifier(n_estimators=100)
        self.binary_classifier = binary_classifier
        self.segment_id = None
        self.kernel=kernel

    @staticmethod
    def core(trajectory, window_size=7, majority_vote_degree=0.9,
             binary_classifier=RandomForestClassifier(n_estimators=100),kernel='Random Walk', verbose=False):

        # cleaned_trajectory = clean_short_trajectories(trajectory, window_size) #assume it is done before calling this class
        e_signal = generate_error_signal(trajectory, window_size,interpolation_name=kernel)

        trajectory_samples = blind_signal_to_sample(e_signal=e_signal, window_size=window_size)

        y_pred = binary_classifier.predict(trajectory_samples)
        y_pred = np.array(
            list([y_pred[0]] * int(window_size / 2)) + list(y_pred) + list([y_pred[-1]] * (int(window_size / 2) + 1)))


        segment_id = majority_vote(y_pred, window_size=window_size, degree=majority_vote_degree)

        # h = harmonic_mean(segments=segment_id, tsid=validation_ground_truth[0], label=validation_ground_truth[1])

        return segment_id

    def tuning(self, tuning_trajectory, tuning_ground_truth=None, window_size_params=None,
               wsii_params=None,
               # majority_vote_degree_params=None, binary_classifier_params=None,
               validation_trajectory=None, validation_ground_truth=None, verbose=False, kernel='Random Walk'):

        if wsii_params is None:
            wsii_params = {'window_size': self.window_size,
                           'majority_vote_degree': self.majority_vote_degree,
                           'binary_classifier': self.binary_classifier,
                           'kernel':self.kernel}

        # if binary_classifier_params == [None]:
        #    binary_classifier_params = [RandomForestClassifier(n_estimators=100)]
        if verbose:
            print("Tuning..")

        if validation_trajectory is None:
            validation_trajectory = tuning_trajectory
            validation_ground_truth = tuning_ground_truth

        assert (len(tuning_ground_truth[0]) == tuning_trajectory.shape[
            0]), "pass the correct labels: length of passed labels is " + str(
            len(tuning_ground_truth[0])) + " and length od X is " + str(tuning_trajectory.shape[0])

        results = {}
        best = None
        best_param = None

        parm_grid = ParameterGrid(wsii_params)
        for p in parm_grid:

            if p['binary_classifier'] is None:
                #train binary classifier using validation
                e_signal = generate_error_signal(validation_trajectory, p['window_size'])
                tsid = validation_ground_truth[0]
                X_train, y_train = signal_to_sample(e_signal=e_signal, tsid=tsid, window_size=p['window_size'])
                p['binary_classifier'] =DecisionTreeClassifier(max_depth=10)
                p['binary_classifier'].fit(X_train, y_train)
                y_pred = p['binary_classifier'].predict(X_train)
                binary_classifier_accuracy = accuracy_score(y_train,y_pred)
                print("binary_classifier_accuracy:", binary_classifier_accuracy * 100.)
            else:

                e_signal = generate_error_signal(validation_trajectory, p['window_size'],interpolation_name=p['kernel'])
                tsid = validation_ground_truth[0]
                X_train, y_train = signal_to_sample(e_signal=e_signal, tsid=tsid, window_size=p['window_size'])
                p['binary_classifier'].fit(X_train, y_train)
                y_pred = p['binary_classifier'].predict(X_train)
                binary_classifier_accuracy = accuracy_score(y_train,y_pred)
                print("binary_classifier_accuracy:", binary_classifier_accuracy * 100.)

            params = {'window_size': p['window_size'],
                      'majority_vote_degree': p['majority_vote_degree'],
                      'binary_classifier': p['binary_classifier'],
                      'kernel':p['kernel']}


            test_param = (p['window_size'], p['majority_vote_degree'], p['binary_classifier'],p['kernel'])
            results[test_param] = self.score(trajectory=tuning_trajectory, ground_truth=tuning_ground_truth,
                                             verbose=verbose, **params)

            print(results[test_param], ":", test_param)
            if best is None:
                best = results[test_param]
                best_param = test_param

            else:
                if best < results[test_param]:
                    best = results[test_param]
                    best_param = test_param

        self.window_size = best_param[0]
        self.majority_vote_degree = best_param[1]
        self.binary_classifier = best_param[2]
        self.kernel=best_param[3]

        # for ws in window_size_params:
        #    for degree in majority_vote_degree_params:
        #        for cls in binary_classifier_params:
        #
        #                    cleaned_trajectory = clean_short_trajectories(tuning_trajectory, ws)  # clean short segments
        #                    tsid = tuning_ground_truth[0]  # tsid##
        #
        #                    print("Quality of dataset:", check_quality_of_trajectory(tuning_ground_truth[0], tuning_ground_truth[1]))
        #                    print("length of training data:", cleaned_trajectory.shape)
        #                    e_signal = generate_error_signal(cleaned_trajectory, ws)
        #                    print("length of error signal:", len(e_signal))#
        #
        #                    X_train, y_train = signal_to_sample(e_signal=e_signal, tsid=tsid, window_size=ws)

        # print(X_train.shape, y_train.shape)

        # rf = b(n_estimators=100)#RandomForestClassifier
        #                    cls.fit(X_train, y_train)

        #                    cleaned_validation_trajectory = clean_short_trajectories(validation_trajectory,
        #                                                                             ws)  # clean short segments
        #                    v_tsid = validation_ground_truth[0]  # tsi

        #                    print("Quality of validation dataset:",
        #                          check_quality_of_trajectory(tuning_ground_truth[0], tuning_ground_truth[1]))
        #                    print("length of training data:", cleaned_validation_trajectory.shape)
        #                    v_e_signal = generate_error_signal(cleaned_validation_trajectory, ws)
        #                    print("length of error signal:", len(v_e_signal))

        #                    X_test, y_test = signal_to_sample(e_signal=v_e_signal, tsid=v_tsid, window_size=ws)

        #                    y_pred = cls.predict(X_test)
        #                    y_pred = np.array(
        #                        list([y_pred[0]] * int(ws / 2)) + list(y_pred) + list([y_pred[-1]] * (int(ws / 2) + 1)))
        #                    binary_classifier_accuracy = cls.score(X_test, y_test)
        #                    print("binary_classifier_accuracy:", binary_classifier_accuracy * 100.)

        #                    segment_id = majority_vote(y_pred, window_size=ws, degree=degree)

        #                    h = harmonic_mean(segments=segment_id, tsid=validation_ground_truth[0],
        #                                      label=validation_ground_truth[1])

        #                    results[(ws, degree, cls)] = h
        #                    if best is None:
        #                        best = results[(ws, degree, cls)]
        #                        best_param = (ws, degree, cls)
        #                    else:
        #                        if best < results[(ws, degree, cls)]:
        #                            best = results[(ws, degree, cls)]
        #                            best_param = (ws, degree, cls)

        #        self.window_size = best_param[0]
        #        self.majority_vote_degree = best_param[1]
        #        self.binary_classifier = best_param[2]
        print("best:", best_param, " with ", best)

        self.segment_id = self.core(trajectory=tuning_trajectory, window_size=self.window_size, majority_vote_degree=self.majority_vote_degree,
                               binary_classifier=self.binary_classifier, verbose=verbose,kernel=kernel)

        return self

    def predict(self, trajectory, verbose=None):
        segment_id = self.core(trajectory,kernel=self.kernel, window_size=self.window_size, majority_vote_degree=self.majority_vote_degree,
                               binary_classifier=self.binary_classifier, verbose=verbose)
        return segment_id

    def score(self, trajectory, ground_truth, verbose=False, **kwargs):
        segment_id = self.core(trajectory, **kwargs)
        h = harmonic_mean(segments=segment_id, tsid=ground_truth[0], label=ground_truth[1])
        if verbose:
            print("Scoring " + ','.join('{0}={1!r}'.format(k, kwargs[k]) for k in kwargs), str(h))
        return h
