from Trajectories.TrajectorySegment import TrajectorySegment
from initialization.NGreedyRandomizedConstruction import build_first_solution
import random, copy
from cost.cost_function import calculate_cost
from optimization.LocalSearchProcedure import optimize_segments
from cost.DistanceMetrics import get_feature_bounds
from old_code.time_calculation import Clock


def grasp_uts_execute(traj, max_distance, min_time, join_coefficient, partitioning_factor, alpha, max_iterations):
    random.seed(a=1, version=2)
    feature_bounds = get_feature_bounds(traj)
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
    c = Clock()
    for i in range(0, max_iterations):
        segments = build_first_solution(traj, max_distance, min_time, partitioning_factor, alpha, random, feature_bounds, landmark_combos_tested)
        segments = optimize_segments(traj,segments, min_time, join_coefficient, partitioning_factor,max_distance,feature_bounds, landmark_combos_tested)
        cost = calculate_cost(segments,max_distance, feature_bounds)
        if cost <= best_cost:
            best_cost = cost
            best_segments = copy.deepcopy(segments)
    #print(str(traj.tid) + ";" + str(len(traj.samplePoints))+ ";" + str(round(best_cost,2)) + ";" + str(is_feasible(best_segments, min_time)))
    return best_segments