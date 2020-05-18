import numpy as np
import math
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.TrajectoryPoint import TrajectoryPoint
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.FeatureBounds import FeatureBounds
def build_point_vector(point, feature_bounds):
    point_vector = []
    feature_names = list(point.trajectoryFeatures.keys())
    feature_names.sort()
    for name in feature_names:
        point_vector.append(normalize(point.trajectoryFeatures[name],feature_bounds[name].min,feature_bounds[name].max))
    return np.array(point_vector)

def distance(point1, point2, feature_bounds):
    #point_vector1 = build_point_vector(point1, feature_bounds)
    #point_vector2 = build_point_vector(point2, feature_bounds)
    return euclidean_distance(point1.point_vector,point2.point_vector)



def cohesiveness(landmark, segment, feature_bounds):
    #landmark_vector = build_point_vector(landmark, feature_bounds)
    landmark_vector = landmark.point_vector
    total = 0
    for key in segment.points:
        p = segment.points[key]
        if landmark.gid != p.gid:
            #point_vector = build_point_vector(p, feature_bounds)
            point_vector = p.point_vector
            total += math.log2(1.+euclidean_distance(landmark_vector, point_vector))
    return total


def cohesiveness2(landmark,segment,feature_bounds):
    landmark_vector = landmark.point_vector

    #    landmark_vector = build_point_vector(landmark, feature_bounds)
    total = 0

    for key in segment.points:
        p = segment.points[key]
        if landmark.gid != p.gid:
#            point_vector = build_point_vector(p, feature_bounds)
            point_vector = p.point_vector

            total += euclidean_distance(landmark_vector, point_vector)
    return total


def euclidean_distance(vector1,vector2):
    return np.linalg.norm(vector1-vector2)
    #return np.sqrt(np.sum(np.power(vector1 - vector2,2)))

def build_avg_vector(segment):
    ar = []

    for p in segment.points:
        ar.append(segment.points[p].point_vector)
    return np.average(ar)

def normalize(value, min, max):
    numerator = value - min
    denominator = max - min
    return numerator / denominator

def get_feature_bounds(trajectory):
    feature_values = {}
    feature_bounds = {}
    feature_names = trajectory.samplePoints[trajectory.firstGid].trajectoryFeatures
    for name in feature_names:
        feature_values[name] = []
    for gid, point in trajectory.samplePoints.items():
        for name in feature_names:
            feature_values[name].append(point.trajectoryFeatures[name])
    for name in feature_names:
        feature_bounds[name] = FeatureBounds(feature_values[name])
    return feature_bounds

