import numpy as np
import math
import sys
import datetime as dt
import time as t
from . import Cost_function as cf

"""
    ----------------------
    Script Description
    ----------------------
    A python script for optimizing the solutions of the RGRASP-SemTS Algorithm. 
    The final output is a multidimensional array with the start and end indexes of the 
    optimized trajectory segments
    
    When using this code, please refer to: 
    
    Junior, Amilcar Soares, Valeria Cesario Times, Chiara Renso, Stan Matwin, and Lucídio AF Cabral. 
    "A semi-supervised approach for the semantic segmentation of trajectories." 
    In 2018 19th IEEE International Conference on Mobile Data Management (MDM), pp. 145-154. IEEE, 2018.    
    
    Original published version: https://ieeexplore.ieee.org/abstract/document/8411271/
    DOI: https://doi.org/10.1109/MDM.2018.00031
    Author version: https://www.researchgate.net/publication/324841537_A_Semi-Supervised_Approach_for_the_Semantic_Segmentation_of_Trajectories
    

"""

"""
    -----------------
    Input parameters
    -----------------
    (1) trajectories_st ([[float, float, string]]) 
            A numpy multidimensional matrix with x_i, y_i, t_i, 
            where x is the trajectory's longitude, y is its latitude, and t is the timestamp 
            as text.
    (2) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (3) min_time (float)
            The minimal time threshold for creating a segment
    (4) semantic_landmarks_pf ([[float,...]])
            A numpy multidimensional matrix with the semantic landmarks point features
    (5) semantic_landmarks_sf[(float),...] 
            A numpy multidimensional matrix with the semantic landmarks segment features            
    (6) max_value (float)
            The maximal similarity value possible between two trajectory points
    (7) segments_class_list [float,...]
            The class value of each segment
    (8) segments_feats_list [(string),...] (optional)
            A list with all segment features attributes to be computed 
    -----------------
    Returns
    -----------------
    (1) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments
    (2) best_cost
            The best solution's cost function value
    (3) final_classes_list_lh
            The classes for each segment found in th optimized solution
    (4) detailed_cost [uns_lh, uns_ldh, sup_lh, sup_ldh]
            The detailed cost of all partial functions
    (5) classes_array ([float,...])
            An array with the class of every segment found by the algorithm

"""

def execute(segments, trajectories_st, trajectories_pf, min_time, semantic_landmarks_pf, semantic_landmarks_sf, max_value, segments_class_list, segments_feats_list):
    
    landmarks_pf = _create_landmarks_pf(segments, trajectories_pf)
    cost, classes_list_lh, detailed_cost, classes_list_ldh, = cf.semi_supervised_cost_function(max_value, trajectories_pf, landmarks_pf, segments, segments_class_list, semantic_landmarks_pf, semantic_landmarks_sf, segments_feats_list)

#   Merge consecutive segments with same landmarks
    segments, classes_array = _merge_segments(segments, segments_class_list)
    
#   Evaluate best position for partitioning the consecutive segments
    for index in range(len(segments)-1):
        s_1, s_2 = segments[index], segments[index+1]
        landmarks_pf = _compute_landmarks_pf(trajectories_pf, s_1, s_2) 
        segments_to_test = np.array([s_1, s_2])
        cost, classes_list_lh, detailed_cost, classes_list_ldh = cf.semi_supervised_cost_function(max_value, trajectories_pf, landmarks_pf, segments_to_test, segments_class_list[index:index+2], semantic_landmarks_pf, semantic_landmarks_sf, segments_feats_list)
        best_solution_set = np.array([[s_1, s_2, cost]])
        pivot = s_1[0] + 1
        # Evaluate every position from pivot, only if solution is feasible regarding time 
        while pivot < segments[index+1][1]-1:
            left_test, right_test = [s_1[0], pivot], [pivot+1, s_2[1]]
            landmarks_pf = _compute_landmarks_pf(trajectories_pf, left_test, right_test) 
            if _is_feasible(trajectories_st, left_test, right_test, min_time):
                segments_to_test = np.array([left_test, right_test])
                cost, c_list_lh, detailed_cost, c_list_ldh = cf.semi_supervised_cost_function(max_value, trajectories_pf, landmarks_pf, segments_to_test, segments_class_list[index:index+2], semantic_landmarks_pf, semantic_landmarks_sf, segments_feats_list)
                best_solution_set = np.concatenate((best_solution_set, [np.array([left_test, right_test, cost])]))
            pivot += 1
        #replace solution by best found in complete set with only feasible solutions
        best_solution_set = best_solution_set[np.argsort(best_solution_set[:, 2])]
        best_solution = best_solution_set[0]
#       replace best solution in segments original set
        segments[index], segments[index+1] = best_solution[0], best_solution[1]

    landmarks_pf = _create_landmarks_pf(segments, trajectories_pf)
    best_cost, final_classes_list_lh, detailed_cost, final_classes_list_ldh  = cf.semi_supervised_cost_function(max_value, trajectories_pf, landmarks_pf, segments, segments_class_list, semantic_landmarks_pf, semantic_landmarks_sf, segments_feats_list)

    return segments, best_cost, final_classes_list_lh, detailed_cost, final_classes_list_ldh, classes_array

"""
    A local method for creating the point features of the landmarks
    -----------------
    Input parameters
    -----------------
    (1) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments        
    (2) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    -----------------
    Returns
    -----------------
    (1) landmarks_pf [[float,...]]
            A numpy multidimensional matrix with the point features of the landmarks

"""
    
def _create_landmarks_pf(segments, trajectories_pf):
    landmarks_pf = np.array([trajectories_pf[segments[0][0]:segments[0][1],:].mean(0)])
    for index in range(1, len(segments)):
        landmarks_pf = np.concatenate((landmarks_pf, [trajectories_pf[segments[index][0]:segments[index][1],:].mean(0)]), axis=0)
    return landmarks_pf
    
"""
    A local method for creating the landmarks point features for two segments, 

    -----------------
    Input parameters
    -----------------
    (1) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (2) left_segment [start_index, end_index]
        The indexes of the left segment
    (3) right_segment [start_index, end_index]
        The indexes of the right segment
    -----------------
    Returns
    -----------------
    (1) landmarks_pf [[float,...],[float,...]]
            A numpy multidimensional matrix with the point features of the landmarks
            for the two segments

"""

def _compute_landmarks_pf(trajectories_pf, left_segment, right_segment):    
    l_mean_arr = trajectories_pf[left_segment[0]:left_segment[1]+1,].mean(0)
    r_mean_arr = trajectories_pf[right_segment[0]:right_segment[1]+1,].mean(0)
    return np.array([l_mean_arr,r_mean_arr])


"""
    A local method for merging consecutive segments with the same class label. 
    -----------------
    Input parameters
    -----------------
    (1) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments  
    (2) class_list [label for s1, label for s2, ...] 
            A list with the labels for the respective segments
    -----------------
    Returns
    -----------------
    (1) merged_segments [[start_index, end_index],...]
            A new list with the segments merged
    (2) class_list [float, ...]
            A new list with the classes of the merged segments

"""

def _merge_segments(segments, class_list):
    merged_segments = None
    min_index = 0
    max_index = 0
    classes_array = []
    if len(class_list) > 1:
        for index in range(len(class_list)-1):
            if class_list[index] == class_list[index+1]:
                max_index = index+1
            else: 
                if merged_segments is None:
                    merged_segments = np.array([[segments[min_index][0], segments[max_index][1]]])
                else:    
                    merged_segments = np.concatenate((merged_segments,[[segments[min_index][0], segments[max_index][1]]]))
                min_index = index+1
                max_index = index+1
                classes_array.append(class_list[index])

    #     add last segment
    if merged_segments is None:
        max_index = len(segments)-1
        merged_segments = np.array([[segments[min_index][0], segments[max_index][1]]])
        classes_array.append(class_list[0])
    else: 
        merged_segments = np.concatenate((merged_segments,[[segments[min_index][0], segments[max_index][1]]]))
    
    if merged_segments.shape[0] != len(classes_array):
        classes_array.append(class_list[len(class_list)-1])
    
    return merged_segments, classes_array

"""
    A local method for verifying if a solution is feasible
    -----------------
    Input parameters
    -----------------
    trajectories_st, left_segment_index, right_segment_index, min_time
    (1) trajectories_st ([[float, float, string]]) 
            A numpy multidimensional matrix with x_i, y_i, t_i, 
            where x is the trajectory's longitude, y is its latitude, and t is the timestamp 
            as text.
    (2) left_segment_index (int)
            The left segment index of a solution
    (3) right_segment_index (int)
            The right segment index of a solution
    (3) min_time
            The minimal time threshold for creating a segment
    -----------------
    Returns
    -----------------
    (1) (Boolean)
            Returns if the solution has the min_time threshold

"""

def _is_feasible(trajectories_st, left_segment_index, right_segment_index, min_time):
    time_diff_left = dt.datetime.strptime(trajectories_st[left_segment_index[1]][2], '%Y-%m-%d %H:%M:%S')-dt.datetime.strptime(trajectories_st[left_segment_index[0]][2], '%Y-%m-%d %H:%M:%S')
    time_diff_right = dt.datetime.strptime(trajectories_st[right_segment_index[1]][2], '%Y-%m-%d %H:%M:%S')-dt.datetime.strptime(trajectories_st[right_segment_index[0]][2], '%Y-%m-%d %H:%M:%S')
    if (time_diff_left.total_seconds() >= min_time) and (time_diff_right.total_seconds() >= min_time):
        return True
    else: return False