import math
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.cost.DistanceMetrics import distance, cohesiveness, cohesiveness2

FEATURE_NAMES = ["speed_inference_m_s","direction_inference"]

def compute_cost(segments, max_distance, feature_bounds):
    return compute_lh(segments, max_distance, feature_bounds) + compute_ldh(segments, feature_bounds)

def compute_lh(segments,max_distance, feature_bounds):
    #return compute_lh2(segments,max_distance,feature_bounds)
    total = 0
    for i in range(0, len(segments) - 1):
        lm1 = segments[i].landmark
        lm2 = segments[i+1].landmark
        total += math.log2(1+max_distance - distance(lm1,lm2, feature_bounds))
    return total

def compute_ldh(segments, feature_bounds):
    #return compute_ldh2(segments,feature_bounds)
    total = 0
    for seg in segments:
        total += cohesiveness(seg.landmark, seg, feature_bounds)
    return total


def compute_lh2(segments,max_distance,feature_bounds):
    total = 0
    for i in range(0, len(segments) - 1):
        lm1 = segments[i].landmark
        lm2 = segments[i + 1].landmark
        total += max_distance - distance(lm1, lm2, feature_bounds)
    return math.log2(1 + total)

def compute_ldh2(segments, feature_bounds):
    total = 0
    for seg in segments:
        total += cohesiveness2(seg.landmark, seg, feature_bounds)
    return math.log2(1+total)
