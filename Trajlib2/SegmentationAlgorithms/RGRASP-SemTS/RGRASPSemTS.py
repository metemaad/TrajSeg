import numpy as np
import sys
import math
from . import Cost_function as cf
from . import GreedyRandomizedConstructionProcedure as build_first_solution
from . import LocalSearchProcedure as local_search
import datetime as dt
import random

"""
    ----------------------
    Algorithm description
    ----------------------
    The Reactive Greedy Randomized Adaptive Search Procedure for semantic Semi-supervised Trajectory 
    Segmentation (RGRASP-SemTS) is an algorithm that segments trajectories combining a limited user 
    labeled dataset with a low number of input parameters and no predefined segmenting criteria.
    
    When using this code, please refer to: 
    
    Junior, Amilcar Soares, Valeria Cesario Times, Chiara Renso, Stan Matwin, and Lucídio AF Cabral. 
    "A semi-supervised approach for the semantic segmentation of trajectories." 
    In 2018 19th IEEE International Conference on Mobile Data Management (MDM), pp. 145-154. IEEE, 2018.    
    
    Original published version: https://ieeexplore.ieee.org/abstract/document/8411271/
    DOI: https://doi.org/10.1109/MDM.2018.00031
    Author version: https://www.researchgate.net/publication/324841537_A_Semi-Supervised_Approach_for_the_Semantic_Segmentation_of_Trajectories
    

"""

"""
    This is the main method of RGRASP-SemTS, and it receives as parameters:
    -----------------
    Input parameters
    -----------------
    (1) seed (int)      
            A seed value for reproducing experiments
    (2) trajectory_data ([[float, ...]...]) 
            A numpy multidimensional matrix with the trajectory data to be segmented.
            It needs to have the following format: 
                [traj_id, x, y, time(string), [traj. features]]
    (3) labeled_dataset. ([[float,...]]) 
            A numpy multidimensional matrix with the labeled trajectory data.
            It needs to have the following format: 
                [traj_id, x, y, time(string), [traj. features], class (string)]
    (5) reactive_proportion (float)
            A reactive proportion value from 0. -- 1. for updating gradually the probabilities
            of the parameters min_time and alpha
    (6) lists_size (float)
            The size of the lists for parameters alpha and min_time
    (7) max_iterations (int)
            The number of max iterations for building and optimizing solutions
    (8) max_value
            The maximal similarity value possible between two trajectory points
    (8) min_time_bounds [min(float), max(float)]
            The min and max value for the parameter min_time in seconds
    (9) segments_feats_list [(string),...] (optional)
            A list with all segment features attributes to be computed 
    -----------------
    Returns
    -----------------
    (1) total_cost (float)
            The total cost of the RGRASP-SemTS function for all traj. segments found
    (2) segment_id_list ([int, ...])
            A list with the segment which every single traj. point was placed
    (3) y_new ([int, ...]) 
            A list with the class with every single traj. point's class
    
"""


def execute(seed, trajectory_data, labeled_dataset, reactive_proportion, lists_size, max_iterations, max_value,
            min_time_bounds, segments_feats_list={'p5', 'p25', 'p50', 'p75', 'p95'}):
    # select unique trajectory ids
    tid_list = np.unique(trajectory_data[:, 0])
    semantic_landmarks_pf, semantic_landmarks_sf, class_name_index, class_proportion = \
        _create_semantic_landmarks(labeled_dataset, segments_feats_list)
    segment_id_index = 1
    segment_id_list = None
    y_new = None
    total_cost = 0.
    for tid in tid_list:
        # slicing the data to get trajectory id (tid), latitude, longitude and time in a numpy array
        trajectory = trajectory_data[trajectory_data[:, 0] == tid]
        trajectory_st = trajectory[:, 1:4]
        trajectory_pf = trajectory[:, 4:].astype(float)

        best_segments, best_solution_cost, best_class_list_lh, class_name_index, best_class_list_ldh, best_class_array = \
            _execute_for_single_trajectory(seed * int(tid), trajectory_st, trajectory_pf, semantic_landmarks_pf,
                                           semantic_landmarks_sf, class_name_index, reactive_proportion, lists_size,
                                           max_iterations,
                                           max_value, min_time_bounds, class_proportion, segments_feats_list)
        total_cost += best_solution_cost
        # TODO too slow, opportunity to paralelize
        for index in range(len(best_class_array)):
            times_to_add = (int(best_segments[index][1]) - int(best_segments[index][0]) + 1)
            arr_sid = np.empty(times_to_add)
            arr_sid.fill(segment_id_index)
            arr_y = np.empty(times_to_add)
            arr_y.fill(best_class_array[index])

            if segment_id_list is None:
                segment_id_list = arr_sid
                y_new = arr_y
            else:
                segment_id_list = np.concatenate((segment_id_list, arr_sid))
                y_new = np.concatenate((y_new, arr_y))

            segment_id_index += 1
    y_new = y_new.astype(int)
    y_pred_cls = []
    for i in y_new:
        # print(i, y_new[i], class_name_index[y_new[i]])
        y_pred_cls.append(class_name_index[i])
    return total_cost, segment_id_list.astype(int), y_pred_cls


def _execute_for_single_trajectory(seed, trajectory_st, trajectory_pf, semantic_landmarks_pf, semantic_landmarks_sf,
                                   class_name_index, reactive_proportion, lists_size, max_iterations, max_value,
                                   min_time_bounds, class_proportion, segments_feats_list):
    # initialize variables constant variables
    random.seed(seed)
    reactive_proportion_val = reactive_proportion * max_iterations

    min_time_list = np.linspace(min_time_bounds[0], min_time_bounds[1], lists_size)
    alpha_list = np.linspace(.1, .6, lists_size)
    min_time_probs = [1. / float(lists_size)] * lists_size
    alpha_probs = [1. / float(lists_size)] * lists_size
    avg_min_time_list = [None] * lists_size
    avg_alpha_list = [None] * lists_size

    best_solution_cost, best_segments, best_class_list = sys.float_info.max, None, None

    # start Reactive GRASP strategy
    for it in range(max_iterations):
        random.seed(seed * it)
        alpha, alpha_index = _weighted_choice(alpha_list, alpha_probs, random)
        random.seed(seed * it * 2)
        min_time, min_time_index = _weighted_choice(min_time_list, min_time_probs, random)
        segments, class_list = build_first_solution.execute(random, trajectory_st, trajectory_pf, min_time, alpha,
                                                            semantic_landmarks_pf, class_proportion)
        optimized_segments, current_cost, current_class_list_lh, detailed_cost, current_class_list_ldh, classes_array = local_search.execute(
            segments, trajectory_st, trajectory_pf, min_time, semantic_landmarks_pf, semantic_landmarks_sf, max_value,
            class_list, segments_feats_list)

        #         swap if best solution so far
        if current_cost < best_solution_cost:
            best_solution_cost = current_cost
            best_segments = optimized_segments
            best_class_list_lh = current_class_list_lh
            best_class_list_ldh = current_class_list_ldh
            best_class_array = classes_array

        #         update average solutions for min_time and alpha lists
        if avg_min_time_list[min_time_index] is None:
            avg_min_time_list[min_time_index] = current_cost
        else:
            avg_min_time_list[min_time_index] = (avg_min_time_list[min_time_index] + current_cost) / 2.

        if avg_alpha_list[alpha_index] is None:
            avg_alpha_list[alpha_index] = current_cost
        else:
            avg_alpha_list[alpha_index] = (avg_alpha_list[alpha_index] + current_cost) / 2.

        #         update weights when reactive proportion is reach
        if (it + 1) % reactive_proportion_val == 0:
            min_time_probs = _update_list_probs(best_solution_cost, avg_min_time_list)
            alpha_probs = _update_list_probs(best_solution_cost, avg_alpha_list)

    return best_segments, best_solution_cost, best_class_list_lh, class_name_index, best_class_list_ldh, best_class_array


"""
    A local method for updating the lists' probabilities of selecting a value. 
    -----------------
    Input parameters
    -----------------
   (1) best_solution_cost (float) 
           The best value of the cost function found
   (2) avg_list [(float),...] 
           A list with all the averages from a list (alpha or min_time)
    -----------------
    Returns
    -----------------
    (1) new_probs [(float),...] 
            A list with the updated new probabilities
"""


def _update_list_probs(best_solution_cost, avg_list):
    all_avg_qi = [0] * len(avg_list)
    for index in range(len(avg_list)):
        if avg_list[index] is not None:
            all_avg_qi[index] = (best_solution_cost / avg_list[index]) ** 10
        else:
            all_avg_qi[index] = best_solution_cost
    total_qi = sum(all_avg_qi)
    new_probs = [all_avg_qi[index] / total_qi for index in range(len(all_avg_qi))]
    return new_probs


"""
    A local method for randomly choosing a value from a list based on its weight.
    -----------------
    Input parameters
    -----------------
    (1) a_list [(float),...] 
            A list with the computed values for 
    (2) weights [(float),...]
            A list with the current weights of the elements of a_list object
    (3) rnd Random 
            The random object to select an element randomly     
    -----------------
    Returns
    -----------------
    (1) element (float)
        The element randomly chosen from the list using the weighted strategy
    (2) index (int)
        The index of the sorted element
"""


def _weighted_choice(a_list, weights, rnd):
    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = rnd.random()
    for index in range(len(weights)):
        if x < weights[index]:
            return a_list[index], index


"""
    A local method for retrieving the minimal time difference between two consecutive trajectory points 
    -----------------
    Input parameters
    -----------------
    (1) time_col [(float),...] 
        A list with all timestamps extracted from a trajectory
    -----------------
    Returns
    -----------------
    (1) The minimal time threshold between consecutive traj points
"""


def _get_min_min_time(time_col):
    min_time = (dt.datetime.strptime(time_col[1], '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime(time_col[0],
                                                                                              '%Y-%m-%d %H:%M:%S')).total_seconds()
    for index in range(1, len(time_col) - 1):
        next_time = (dt.datetime.strptime(time_col[index + 1], '%Y-%m-%d %H:%M:%S') - dt.datetime.strptime(
            time_col[index], '%Y-%m-%d %H:%M:%S')).total_seconds()
        if next_time < min_time:
            min_time = next_time
    return min_time


"""
    A local method for creating the semantic landmarks from the labeled dataset
    -----------------
    Input parameters
    -----------------
    (1) labeled_dataset [(float),...] 
        A list with all timestamps extracted from a trajectory
    (2) segments_feats_list [(string),...] 
        A list with all segment features attributes to be computed
    -----------------
    Returns
    -----------------
    (1) semantic_landmarks_pf [(float),...]
            A numpy multidimensional matrix with the semantic landmarks point features
    (2) semantic_landmarks_sf[(float),...] 
            A numpy multidimensional matrix with the semantic landmarks segment features
"""


def _create_semantic_landmarks(labeled_dataset, segments_feats_list):
    classes_col = labeled_dataset[:, -1]
    class_proportion = []
    # deleting the spatio-temporal columns
    tids = labeled_dataset[:, 0]
    tids = tids[:, np.newaxis]
    pfs = labeled_dataset[:, 4:]

    # _labeled_dataset = labeled_dataset[:,4:]

    _labeled_dataset = np.concatenate((tids, pfs), axis=1)
    classes = np.sort(np.unique(classes_col))
    semantic_landmarks_pf, semantic_landmarks_sf = None, None
    class_name_index = []
    for item in classes:
        #       select trajectory points from a single class
        idx_for_class = (_labeled_dataset[:, -1] == item)
        all_tps = _labeled_dataset[idx_for_class]
        all_tps = np.delete(all_tps, -1, axis=1)
        class_name_index.append(item)
        #       compute the point and segment features
        # iterate over segments
        sf_for_tids = None
        for tid in np.unique(all_tps[:, 0]):
            idx_for_tid = (all_tps[:, 0] == tid)
            all_tps_for_tid = all_tps[idx_for_tid]
            all_tps_for_tid = np.delete(all_tps_for_tid, 0, axis=1)
            v = cf._calculate_all_sf(all_tps_for_tid, segments_feats_list)
            if sf_for_tids is None:
                sf_for_tids = np.array([v])
            else:
                sf_for_tids = np.concatenate((sf_for_tids, [v]))

        all_sf = cf._calculate_all_sf(all_tps[:, 1:], segments_feats_list)
        tps_mean = all_tps[:, 1:].mean(0)
        all_sf = sf_for_tids.mean(0)

        if semantic_landmarks_pf is None and semantic_landmarks_sf is None:
            semantic_landmarks_pf = np.array([tps_mean])
            semantic_landmarks_sf = np.array([all_sf])
        else:
            semantic_landmarks_pf = np.concatenate((semantic_landmarks_pf, [tps_mean]))
            semantic_landmarks_sf = np.concatenate((semantic_landmarks_sf, [all_sf]))
        class_proportion.append(float(all_tps.shape[0]) / float(pfs.shape[0]))

    return semantic_landmarks_pf, semantic_landmarks_sf, class_name_index, class_proportion
