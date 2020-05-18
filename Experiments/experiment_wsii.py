import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

import Trajlib2.SegmentationEvaluation as SegmentationEvaluation
from Experiments.Fishing.plot import plot_segments
from Trajlib2.TrajectorySegmentation import TrajectorySegmentation
from Trajlib2.core.traj_reconstruct import plot_df

class WSIIExperiment:
    def __init__(self, __datafile__, __algorithm__, __dataset_name__, __load_data_function__,
                 __plotting__, __seed__,
                 __tuning_parameters__, __verbose__=False):
        self.process = psutil.Process(os.getpid())
        self.track_memory = []
        self.start_time = time.process_time()
        self.verbose = __verbose__
        self.__plotting__ = __plotting__
        self.__datafile__ = __datafile__
        self.__algorithm__ = __algorithm__
        self.__dataset_name__ = __dataset_name__
        self.__load_data_function__ = __load_data_function__
        self.__seed__ = __seed__
        self.__tuning_parameters__ = __tuning_parameters__
        self.__Experiment_Name__ = self.__dataset_name__ + "_" + self.__algorithm__
        self.ds = None
        try:
            os.mkdir(self.__dataset_name__)

        except FileExistsError:

            pass

    def execute(self):
        random.seed(self.__seed__)

        print(time.process_time() - self.start_time)
        print(" Memory usage:", self.process.memory_info().rss / 1000000, "MB")

        self.track_memory.append(['start', time.process_time(), self.process.memory_info().rss / 1000000])

        print(pd.DataFrame(self.track_memory, columns=['milestone', 'time', 'memory']))

        self.track_memory.append(['loading data', time.process_time(), self.process.memory_info().rss / 1000000])
        self.ds = self.__load_data_function__(self.__datafile__)
        self.track_memory.append(['data loaded', time.process_time(), self.process.memory_info().rss / 1000000])
        h = self.__main__(**self.__tuning_parameters__, verbose=self.verbose)
        return h

    @staticmethod
    def prepare_test_tun_sets(ds, i):
        listOfDatasets = set(range(len(ds)))
        tuning = ds[i]
        test = pd.DataFrame()
        for k in listOfDatasets - set([i]):
            test = test.append(ds[k])
        test = test.copy()
        tuning = tuning.loc[~tuning.index.duplicated(keep='first')]

        test = test.loc[~test.index.duplicated(keep='first')]



        tuning = tuning.sort_index()
        test = test.sort_index()
        return test ,tuning

    @staticmethod
    def tuning_step(tuning, **kwargs):
        s = TrajectorySegmentation()
        tsid = np.array(tuning.loc[:, ['TSid']].values.ravel())
        label = np.array(tuning.loc[:, ['label']].values.ravel())
        seg_id, best_parameters = s.segment_by_wsii(tuning_set=tuning, tuning_set_ground_truth=(tsid, label), **kwargs)
        return best_parameters, s

    @staticmethod
    def testing(test_trajectory, s, best_parameters):
        s = TrajectorySegmentation(test_trajectory)
        seg_id, best_parameters = s.segment_by_wsii(**best_parameters, verbose=False)

        return seg_id, best_parameters

    def reporting(self, testing_set, best_parameters, i):


        print("Reporting fold:", i)
        TSid = np.array(testing_set.loc[:, ['TSid']].values.ravel())
        label = np.array(testing_set.loc[:, ['label']].values.ravel())
        segments = np.array(testing_set.loc[:, ['segment_id']].values.ravel())
        plot_df(testing_set, TSid, segments)
        if self.__plotting__:
            plot_segments(testing_set, segments, i)
        r = SegmentationEvaluation.report(segments=segments, label=label, tsid=TSid, verbose=False)
        print("----", "Harmonic mean:", r[2])
        p, c, h, k_segment1, k_segment2, k_tsid, acc, kappa, pr, re, j, p2, c2, h2, homo, comp, v_measure = r
        results = [p * 100, c * 100, h * 100, best_parameters, k_segment1, k_segment2, k_tsid, acc * 100, kappa * 100,
                   pr * 100, re * 100, j * 100, p2 * 100, c2 * 100, h2 * 100, homo * 100, comp * 100, v_measure * 100]
        print("Number of segments:", k_tsid, "Number of generated segments:", k_segment1, k_segment2)
        return results

    def save_results(self, results_of_evaluation, track_memory_array):

        df = pd.DataFrame(results_of_evaluation, columns=['Purity', 'Coverage', 'Harmonic mean', 'best parameters','k_segment1', 'k_segment2', 'k_tsid', 'acc', 'kappa', 'pr', 're', 'j', 'p2', 'c2', 'h2','homo','comp','v_measure'])
        df.to_csv(self.__dataset_name__ + "/Results_" + self.__Experiment_Name__ + ".csv")
        df.boxplot()
        plt.title(self.__Experiment_Name__)
        plt.savefig(self.__dataset_name__ + "/Results_" + self.__Experiment_Name__ + ".png")
        plt.show()
        pd.DataFrame(track_memory_array, columns=['milestone', 'time', 'memory']).to_csv(
            self.__dataset_name__ + '/Performance_' + self.__Experiment_Name__ + ".csv")
        print(pd.DataFrame(track_memory_array, columns=['milestone', 'time', 'memory']))
        print(df)

        print('Harmonic mean:', df['Harmonic mean'].mean())
        print('Purity:', df['Purity'].mean())
        print('Coverage:', df['Coverage'].mean())

    def __main__(self, **kwargs):
        results_of_evaluation = []
        listOfDatasets = set(range(len(self.ds)))

        for i in listOfDatasets:
            self.track_memory.append(
                ['Start preparing fold' + str(i), time.process_time(), self.process.memory_info().rss / 1000000])
            pd.DataFrame(self.track_memory, columns=['milestone', 'time', 'memory']).to_csv(
                self.__dataset_name__ + '/Performance_' + self.__Experiment_Name__ + ".csv")
            print("----", i, " Memory usage:", self.process.memory_info().rss / 1000000, "MB")
            exp_time = time.process_time()

            test, tuning = self.prepare_test_tun_sets(self.ds, i)
            print("----", "train :", tuning.shape, "test:", test.shape)

            # create segmentation object for test data
            if self.__plotting__:
                plt.scatter(tuning.lat, tuning.lon)
                plt.show()
                plt.scatter(test.lat, test.lon, c='r')
                plt.show()

            print("----", "Start tuning...")
            self.track_memory.append(
                ['Start tuning' + str(i), time.process_time(), self.process.memory_info().rss / 1000000])
            tuning_time = time.process_time()

            best_parameters, segmentation_algorithm = self.tuning_step(tuning, **kwargs)

            print("----", "best param", best_parameters)
            print("----", "Tuning time:", time.process_time() - tuning_time, "seconds")
            self.track_memory.append(
                ['End tuning' + str(i), time.process_time(), self.process.memory_info().rss / 1000000])
            print("----", "test folds ", list(listOfDatasets - set([i])), ":", test.shape, "tuning fold {", i, "}:",
                  tuning.shape)
            test_time = time.process_time()

            # Test the algorithm
            seg_id, best_parameters = self.testing(test, segmentation_algorithm, best_parameters)
            test = test.assign(segment_id=seg_id)

            print("----", "Test time:", time.process_time() - test_time, "seconds")
            self.track_memory.append(
                ['end testing' + str(i), time.process_time(), self.process.memory_info().rss / 1000000])

            # report results of testing
            result_data = self.reporting(test, best_parameters, i)
            results_of_evaluation.append(result_data)

            print("----", "experiment time:", time.process_time() - exp_time, "seconds")
            self.track_memory.append(
                ['end reporting' + str(i), time.process_time(), self.process.memory_info().rss / 1000000])

        self.save_results(results_of_evaluation, self.track_memory)
        print("----", 'total:', time.process_time() - self.start_time, "seconds")
        self.track_memory.append(
            ['end Experiment' + str(self.__Experiment_Name__), time.process_time(),
             self.process.memory_info().rss / 1000000])
        df = pd.DataFrame(results_of_evaluation, columns=['Purity', 'Coverage', 'Harmonic mean', 'best parameters','k_segment1', 'k_segment2', 'k_tsid', 'acc', 'kappa', 'pr', 're', 'j', 'p2', 'c2', 'h2','homo','comp','v_measure'])
        print('Harmonic mean:', df['Harmonic mean'].mean())
        return df['Harmonic mean']
