from Trajlib2.SegmentationAlgorithms.GRASP_UTS.cost.DistanceMetrics import distance, build_avg_vector, build_point_vector,euclidean_distance
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.BuildSegments.BuildSegments import build_segments
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Feasibility import is_feasible
import copy

def optimize_segments(traj,segments, min_time, join_coefficient, partioning_factor, max_distance,feature_bounds, landmark_combos_tested):
    distance_threshold = join_coefficient * max_distance
    landmarks = []
    best_segments = copy.deepcopy(segments)
    for s in segments:
        landmarks.append(s.landmark.copy())
    i=0
    while i < len(landmarks)-1:
        landmark1 = landmarks[i]
        landmark2 = landmarks[i+1]
        d = distance(landmark1, landmark2, feature_bounds)
        if d <= distance_threshold:
            del landmarks[i+1]
            i -= 1
        i += 1
    if len(landmarks)<=1:
        return best_segments
    segments = build_segments(traj,landmarks,partioning_factor, landmark_combos_tested,True,max_distance,min_time, feature_bounds)
    #lms = []
    #Just Changed this to test the results
    if is_feasible(segments, min_time):  # java code says to execute this section if not feasible but that doesn't make sense?
        for s in segments:
            best_distance = float('inf')
            avg_vector = build_avg_vector(s)
            for i in s.points:
                #p_vector = build_point_vector(s.points[i],feature_bounds)
                p_vector = s.points[i].point_vector
                dist = euclidean_distance(avg_vector,p_vector)
                if dist< best_distance:
                    best_distance = dist
                    s.landmark = s.points[i]
            #lms.append(s.landmark)
                #curr_distance = 0
                #for j in s.points:
                 #   if i == j:
                 #       continue
                 #   curr_distance += distance(s.points[i],s.points[j],feature_bounds)
                #if curr_distance < best_distance:
                #    best_distance = curr_distance
        #Just Added This
        #segments = build_segments(traj,lms,partioning_factor,landmark_combos_tested,False,max_distance,min_time,feature_bounds)
        return segments
    else:
        return best_segments
