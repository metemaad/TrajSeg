from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.TrajectorySample import TrajectorySample
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.TrajectoryPoint import TrajectoryPoint
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.TrajectorySegment import TrajectorySegment
from datetime import datetime
import numpy as np
import random
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.initialization.NGreedyRandomizedConstruction import build_first_solution
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.optimization.LocalSearchProcedure import optimize_segments
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.cost.cost_function import calculate_cost
import copy, math
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.cost.DistanceMetrics import get_feature_bounds, build_point_vector


class GRASPUTS():
    def __init__(self, feature_names, min_time=6, join_coefficient=0.3, partitioning_factor=0,
                 alpha=0.3, lat_col='lat', lon_col='lon', dt_col='time', time_format='', max_iter=30, normalize=True):
        self.parameters = {}
        self.parameters['min_time'] = min_time
        self.parameters['join_coefficient'] = join_coefficient
        self.parameters['partitioning_factor'] = partitioning_factor
        self.parameters['alpha'] = alpha
        self.max_iter = max_iter
        self.lat = lat_col
        self.lon = lon_col
        self.time = dt_col
        self.feature_names = feature_names
        self.time_format = time_format
        self.normalize = normalize

    def segment(self, df, max_distance=None):
        df = df.copy()
        if self.normalize:
            for feature in self.feature_names:
                df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
        traj = self.__build_traj(df)
        if max_distance is None:
            max_distance = math.sqrt(len(self.feature_names))
        return self.__execute(traj, max_distance)

    def __build_traj(self, df):
        count = 0
        traj = TrajectorySample()
        for idx, row in df.iterrows():
            point = TrajectoryPoint()
            point.tid = 0
            traj.tid = 0
            point.gid = count
            count += 1
            point.longitude = row[self.lon]
            point.latitude = row[self.lat]

            #point.timestamp = datetime.strptime(str(row[self.time]), self.time_format)
            point.timestamp = datetime.strptime(str(idx), self.time_format)
            point_vector = {}
            act_vector = []
            for feature in self.feature_names:
                act_vector.append(row[feature])

            # for feature in self.feature_names:
            #    point_vector[feature] = row[feature]
            point.trajectoryFeatures = point_vector
            traj.samplePoints[point.gid] = point
            traj.samplePoints[point.gid].point_vector = np.array(act_vector)
        traj.lastGid = count - 1
        # for k in traj.samplePoints.keys():
        #    print(k)
        #    traj.samplePoints[k].point_vector = build_point_vector(traj.samplePoints[k],get_feature_bounds(traj))
        return traj

    def __execute(self, traj, max_dist):
        #    feature_bounds = get_feature_bounds(traj)
        if len(traj.samplePoints) <= 2:
            s = TrajectorySegment()
            s.points = traj.samplePoints
            s.landmark = traj.firstGid
            segments = []
            segments.append(s)
            return segments
        best_segments = []
        best_cost = float('inf')
        landmark_combos_tested = {}
        feat_bounds = get_feature_bounds(traj)

        for i in range(0, self.max_iter):
            # print("Iteration",i)
            segments = build_first_solution(traj, max_dist, self.parameters['min_time'],
                                            self.parameters['partitioning_factor'],
                                            self.parameters['alpha'], random,
                                            feat_bounds,
                                            landmark_combos_tested)
            segments = optimize_segments(traj, segments, self.parameters['min_time'],
                                         self.parameters['join_coefficient'],
                                         self.parameters['partitioning_factor'],
                                         max_dist,
                                         feat_bounds,
                                         landmark_combos_tested)
            cost = calculate_cost(segments, max_dist, feat_bounds)
            # print("Iteration",i,"Cost:",cost)
            if cost <= best_cost:
                best_cost = cost
                best_segments = copy.deepcopy(segments)
        # print("Segments",len(best_segments),"Best Cost",best_cost)
        return best_segments, best_cost


class GRASPUTSEvaluate:
    @staticmethod
    def get_ground_truth(df, label):
        count = 0
        ground_truth = []
        last = df.iloc[0][label]
        for idx, row in df.iterrows():
            if row[label] != last:
                count += 1
            ground_truth.append(count)
            last = row[label]
        return ground_truth

    @staticmethod
    def get_predicted(segs):
        prediction = []
        count = 0
        for s in segs:
            s.compute_segment_features()
            prediction += [count] * (s.lastGid - s.firstGid + 1)
            count += 1
        return prediction

    @staticmethod
    def purity(ground_truth, labels):
        avg = []
        ground_truth = np.array(ground_truth)
        labels = np.array(labels)
        for ts in set(labels):
            ma = 0
            g = ground_truth[(np.where(labels == ts)[0])]
            for tp in set(g):
                _ = len(np.where(g == tp)[0])
                if _ > ma:
                    ma = _
            if ts != -1:
                avg.append(ma * 1.0 / len(g))
        return avg, np.mean(np.array(avg))

    @staticmethod
    def coverage(ground_truth, labels):
        cov = []
        labels = np.array(labels)
        ground_truth = np.array(ground_truth)
        for ts in set(ground_truth):
            mx = 0
            g = labels[(np.where(ground_truth == ts)[0])]
            for l in set(g):
                _ = len(np.where(g == l)[0])
                if mx <= _:
                    mx = _
            cov.append(mx * 1.0 / len(g))
        return cov, np.mean(np.array(cov))

    @staticmethod
    def harmonic_mean(ground_truth, prediction):
        cov = GRASPUTSEvaluate.coverage(ground_truth, prediction)[1]
        pur = GRASPUTSEvaluate.purity(ground_truth, prediction)[1]
        return (2 * cov * pur) / (cov + pur)
