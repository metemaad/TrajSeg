from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.TrajectorySegment import TrajectorySegment
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.cost.cost_function import calculate_cost
from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Feasibility import is_feasible


def build_segments(traj, landmarks, partioning_factor, landmark_combos_tested, put_in_map, max_distance, min_time, feature_bounds):
    if len(landmarks) == 0:
        return None
    landmarks.sort(key=lambda x : x.gid, reverse=False)
    segments = []
    seg_ids = 1

    s = TrajectorySegment()
    s.landmark = landmarks[0]
    s.segmentId = seg_ids
    i = traj.firstGid # This is equivalent to the java code?

    while(i<landmarks[0].gid):
        s.points[i] = traj.samplePoints[i]
        i += 1
    segments.append(s)

    for i in range(0,len(landmarks)-1):
        seg_ids+=1
        landmark1 = landmarks[i].gid
        landmark2 = landmarks[i+1].gid
        complex_list_key = str(landmark1)+"_"+str(landmark2)
        first_segment = segments[len(segments) - 1]
        second_segment = TrajectorySegment()
        second_segment.segmentId = seg_ids
        second_segment.landmark = landmarks[i + 1]
        second_segment.points = {}

        if complex_list_key in landmark_combos_tested:
            binary_result = landmark_combos_tested[complex_list_key]
        else:
            increment = landmark1
            sublist = {}
            while increment <= landmark2:
                sublist[increment] = traj.samplePoints[increment]
                increment += 1
            seg_ids += 1
            binary_result = partitioning_search(first_segment,second_segment,sublist,partioning_factor, max_distance, min_time, feature_bounds)
            if put_in_map:
                landmark_combos_tested[complex_list_key] = binary_result

        first_list = binary_result[0]
        second_list = binary_result[1]

        for gid in first_list:
            first_segment.points[gid] = first_list[gid]

        for gid in second_list:
            second_segment.points[gid] = second_list[gid]
        segments.append(second_segment)
    last_segment = segments[len(segments)-1]
    i = last_segment.landmark.gid
    while i <= traj.lastGid:
        last_segment.points[i] = traj.samplePoints[i]
        i += 1
    return segments


def partitioning_search(first_segment, second_segment, sublist, partitioning_factor, max_distance, min_time, feature_bounds):
    if len(sublist)==0:
        return [{},{}]

    keys = list(sublist.keys())
    keys.sort()

    counter = round(partitioning_factor*len(keys))

    if counter == 0:
        counter = 1

    i = keys[0]
    first_list = {sublist[i].gid: sublist[i]}
    second_list = {}

    j = i+1

    while j<keys[len(keys)-1]:
        second_list[j] = sublist[j]
        j += 1

    best_cost = float('inf')
    first_segment_evaluate = TrajectorySegment()
    second_segment_evaluate = TrajectorySegment()
    first_segment_evaluate.points = first_list
    second_segment_evaluate.points = second_list
    first_segment_evaluate.landmark = first_segment.landmark
    second_segment_evaluate.landmark = second_segment.landmark
    best_list_one = {}
    best_list_two = {}
    i += 1
    best_i = keys[0]
    while i < keys[len(keys)-1]:
        if is_feasible([first_segment_evaluate,second_segment_evaluate],min_time):
            cost = calculate_cost([first_segment_evaluate,second_segment_evaluate], max_distance, feature_bounds)
            if cost < best_cost:
                best_cost = cost
                best_i = i
        first_list[i] = sublist[i]
        del second_list[i]
        i += counter
    j=keys[0]
    while j<keys[len(keys)-1]:
        if j<=best_i:
            best_list_one[j] = sublist[j]
        else:
            best_list_two[j] = sublist[j]
        j+=1
    best_list = [best_list_one, best_list_two]

    return best_list