from Trajlib2.SegmentationAlgorithms.GRASP_UTS.BuildSegments.BuildSegments import build_segments
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Feasibility import is_feasible
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.cost.DistanceMetrics import distance
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.TrajectorySegment import TrajectorySegment
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.old_code.time_calculation import Clock
def build_first_solution(traj, max_distance, min_time, partitioning_factor, alpha, random, feature_bounds, landmark_combos_tested):
    keys = []
    c = Clock()
    for key in traj.samplePoints:
        keys.append(key)
    keys.sort()
    feasible_segments = []
    chosen_landmarks = []
    sorted_position = random.randint(0,len(keys)-1)
    sorted_gid = keys[sorted_position]
    sorted_point = traj.samplePoints[sorted_gid].copy()
    while len(keys) != 0:
        point_in_rcl = build_rcl(keys, traj.samplePoints,alpha)
        if len(feasible_segments) >= 1:
            if len(point_in_rcl) > 1:
                sorted_position = random.randint(0, len(point_in_rcl) - 1)
            else:
                sorted_position = 0
            sorted_point = point_in_rcl[sorted_position]
        chosen_landmarks.append(sorted_point)
        segments = build_segments(traj, chosen_landmarks,partitioning_factor,landmark_combos_tested,True,max_distance,min_time,feature_bounds)
        keys.remove(sorted_point.gid)
        if not is_feasible(segments,min_time):
            chosen_landmarks.remove(sorted_point)
            point_in_rcl.remove(sorted_point)
        else:
            feasible_segments = segments
            keys = remove_keys_in_neighborhood_of_segment(sorted_point,feasible_segments,keys,min_time,traj,feature_bounds)
            keys = sort_keys(sorted_point, keys, traj.samplePoints,feature_bounds)


    return feasible_segments


def remove_keys_in_neighborhood_of_segment(sorted_point, feasible_segments, keys, min_time, traj, feature_bounds):
    keys.sort()

    seg = find_segment(sorted_point,feasible_segments)
    min = seg.firstGid
    max = seg.lastGid
    curr = sorted_point.gid
    right = curr + 1
    left = curr - 1
    s = TrajectorySegment()
    s.points[curr] = sorted_point
    while not is_feasible([s],min_time):
        if right > max:
            s.points[left] = traj.samplePoints[left].copy()
            left -= 1
        elif left < min:
            s.points[right] = traj.samplePoints[right].copy()
            right += 1
        else:
            right_cost = distance(traj.samplePoints[right],sorted_point, feature_bounds)
            left_cost = distance(traj.samplePoints[left], sorted_point, feature_bounds)
            if right_cost < left_cost:
                s.points[right] = traj.samplePoints[right].copy()
                right += 1
            else:
                s.points[left] = traj.samplePoints[left].copy()
                left -= 1
    for p in s.points:
        for k in keys:
            if k == p:
                keys.remove(k)
                break
    return keys


def find_segment(sorted_point, feasible_segments):
    for s in feasible_segments:
        s.compute_segment_features()
        if sorted_point.gid >= s.firstGid and sorted_point.gid <= s.lastGid:
            return s
    return None


def build_rcl(keys, sample_points, alpha):
    rcl = []
    if len(keys) >= 1:
        new_list_size = round(alpha * len(keys))
        if new_list_size <=1:
            rcl.append(sample_points[keys[0]])
        else:
            i=0
            while i < new_list_size and i < len(keys):
                rcl.append(sample_points[keys[i]])
                i+=1
    else:
        rcl.append(sample_points[keys[0]])
    return rcl


def sort_keys(sorted_point, keys, sample_points, point_boundaries):
    distance_list = []
    for gid in keys:
        point_to_add = sample_points[gid]
        distance_list.append((distance(sorted_point, point_to_add, point_boundaries), point_to_add))
    distance_list.sort(key=lambda x : x[0])
    new_keys = []
    for tup in distance_list:
        new_keys.append(tup[1].gid)
    return new_keys
