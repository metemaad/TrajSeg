import numpy as np
import math
import sys
import datetime as dt
import time as t
from datetime import timedelta
from . import Cost_function as cf


"""
    ----------------------
    Script Description
    ----------------------
    A python script for building the first feasible solution for the RGRASP-SemTS Algorithm. 
    The final output is a multidimensional array with the start and end indexes of the 
    trajectory segments
    
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
    (1) rnd (Random)
            The random object for selecting trajectory points as landmarks candidates
    (2) trajectories_st ([[float, float, string]]) 
            A numpy multidimensional matrix with x_i, y_i, t_i, 
            where x is the trajectory's longitude, y is its latitude, and t is the timestamp 
            as text.
    (3) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (4) min_time
            The minimal time threshold for creating a segment
    (5) alpha
            The alpha value for creating the restricted candidates list (RCL)
    (6) semantic_landmarks_pf ([[float,...]])
            A numpy multidimensional matrix with the semantic landmarks point features
    (7) 
    -----------------
    Returns
    -----------------
    (1) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments
    
"""

def execute(rnd, trajectories_st, trajectories_pf, min_time, alpha, semantic_landmarks_pf, class_proportion):
    #initialize indexes list [traj_point_index, class_index, min_similarity]
    candidates_list = _compute_candidates_cost(trajectories_pf, semantic_landmarks_pf)
    #sort cadidates by min_similarity and reduce the RCL list size by alpha factor
    full_list = candidates_list[np.argsort(candidates_list[:, 2])]
    RCL_list = _build_balanced_RCL(full_list, class_proportion, alpha) 
    RCL_list = RCL_list[np.argsort(RCL_list[:, 2])]
    
    list_of_used_points = np.array([])
    first_complete_solution = np.array([])
    class_list = []
    #Iterate over RCL list while candidates are available
    while RCL_list.size > 0: 
        list_of_used_points = np.sort(list_of_used_points, axis=0)
        #get a random trajectory point as a candidate, and verify if it was used in any segment 
        candidate_pos = rnd.randrange(RCL_list.shape[0])
        candidate = RCL_list[candidate_pos]
        #delete if candidate is not feasible, or proceed otherwise
        if candidate[0] in list_of_used_points:
            RCL_list = np.delete(RCL_list, candidate_pos, 0)
            continue
        if list_of_used_points.size > 0 and candidate[0] not in list_of_used_points: 
            list_of_used_points = np.append(list_of_used_points, candidate[0])
        else: 
            list_of_used_points = np.array(candidate[0])
        RCL_list = np.delete(RCL_list, candidate_pos, 0)
        
        #grow the segment in the best direction, if possible         
        left_index, right_index = int(candidate[0])-1, int(candidate[0])+1
        landmark_pf = trajectories_pf[int(candidate[0])]
        new_segment, is_feasible, list_of_used_points = _decide_direction(np.array([int(candidate[0]),int(candidate[0])]), landmark_pf, left_index, right_index, trajectories_pf, semantic_landmarks_pf, int(candidate[1]), list_of_used_points)
        if not is_feasible:
            continue
        time_diff = dt.datetime.strptime(trajectories_st[new_segment[1]][2], '%Y-%m-%d %H:%M:%S')-dt.datetime.strptime(trajectories_st[new_segment[0]][2], '%Y-%m-%d %H:%M:%S')
        while  time_diff.total_seconds() < min_time: 
            #Grow solution until min_time threshold is satisfied
            left_index, right_index = new_segment[0]-1, new_segment[1]+1
            landmark_pf =  trajectories_pf[new_segment[0]:new_segment[1],:].mean(0)#matrix.mean(0)
            new_segment, is_feasible, list_of_used_points = _decide_direction(new_segment, landmark_pf, left_index, right_index, trajectories_pf, semantic_landmarks_pf, int(candidate[1]), list_of_used_points)
            
            #keep growing if possible, but stop if there's no direction to grow the segment
            if is_feasible:
                time_diff = dt.datetime.strptime(trajectories_st[new_segment[1]][2], '%Y-%m-%d %H:%M:%S')-dt.datetime.strptime(trajectories_st[new_segment[0]][2], '%Y-%m-%d %H:%M:%S')
            else: break
        
        #only add the previous solution if min_time condition was satisfied
        if is_feasible:
            if first_complete_solution.size > 0:
                first_complete_solution = np.concatenate((first_complete_solution, [new_segment]), axis=0)
            else: 
                first_complete_solution = np.array([new_segment])
            class_list.append(candidate[1])
        #sort all solutions and list of used points (points that are not feasible or previously used in one solution)        
        first_complete_solution = np.sort(first_complete_solution, axis=0)
        list_of_used_points = np.sort(list_of_used_points)
        
    if len(class_list) == 0:
        class_list.append(0.)
        
    return _distribute_remaining_tpoints(first_complete_solution, trajectories_pf.shape[0]-1), class_list


"""
    -----------------
    Input parameters
    -----------------
    
    (1) full_list 
            A multidimensional array with the the traj point index, 
            the class index for the traj point, and the similarity value
    (2) class_proportion [float, ...]
            The percentage of the proportion of each class in the labeled dataset
    (3) alpha
            The alpha value for creating the restricted candidates list (RCL)
    
    -----------------
    Returns
    -----------------
    (1) rcl
            A numpy multidimensional matrix with 
    
"""

def _build_balanced_RCL(full_list, class_proportion, alpha):
    full_size = int(alpha*len(full_list))
    rcl = None
    for index in range(len(class_proportion)):
        idx_for_class = (full_list[:, 1] == index)
        sub_list = full_list[idx_for_class]
        n_get = int(full_size*class_proportion[index])
        if n_get > 0:
            c_elements = sub_list[0:n_get,:]
        else:
            c_elements = sub_list[0:1,:]
        if rcl is None: 
            rcl = np.array(c_elements)
        else: 
            rcl = np.concatenate((rcl, c_elements))

    return rcl


"""
    This function adds the remaining points that were not placed in any solution.
    -----------------
    Input parameters
    -----------------
    (1) first_complete_solution 
            The first complete feasible solution build for the given parameters
    (2) last_index (float)
            The last index value of the trajectory points
    -----------------
    Returns
    -----------------    
    (1) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments
    
"""

def _distribute_remaining_tpoints(first_complete_solution, last_index):
    for index in range(len(first_complete_solution)-1):
        indexes_to_distribute = first_complete_solution[index+1][0] - first_complete_solution[index][1] - 1
        if indexes_to_distribute != 0:
            if indexes_to_distribute == 1:
                first_complete_solution[index][1] = first_complete_solution[index][1]+1
            else: 
                increment = int(indexes_to_distribute/2)
                first_complete_solution[index][1] = first_complete_solution[index][1]+increment
                first_complete_solution[index+1][0] = first_complete_solution[index+1][0]-increment
                if indexes_to_distribute % 2 != 0:
                    first_complete_solution[index][1] = first_complete_solution[index][1]+1
    if first_complete_solution.shape[0] == 0:
        first_complete_solution = np.array([[0, last_index]])
    else: 
        first_complete_solution[0][0] = 0
        first_complete_solution[len(first_complete_solution)-1][1] = last_index
    return first_complete_solution


"""
    This function decides the direction to grow a segment in the best direction 
    (according to RGRASP-SemTS cost function) possible.
    -----------------
    Input parameters
    -----------------
    (1) segment [start_index, end_index]
            The indexes of a segment
    (2) landmark_pf [float,...]
            A numpy multidimensional matrix with the point features of the landmark
    (3) left_index
            The left index value of a segment
    (4) right_index
            The right index value of a segment
    (5) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (6) semantic_landmarks_pf ([[float,...]])
            A numpy multidimensional matrix with the semantic landmarks point features
    (7) list_of_used_points [int,...]
            the list of used points updated if a direction was taken
    -----------------
    Returns
    -----------------
    (1) segment [start_index, end_index]
            the final segment with the updated direction, 
    (2) is_feasible (Boolean)
            a flag with true if there was a direction to grow, and false otherwise
    (3) list_of_used_points [int,...]
            the list of used points updated if a direction was taken    
"""

def _decide_direction(segment, landmark_pf, left_index, right_index, trajectories_pf, semantic_landmarks_pf, semantic_landmark_index, list_of_used_points):
    new_segment = np.array([[]])
    is_feasible = True
    if left_index not in list_of_used_points and right_index not in list_of_used_points:
        direction = _eval_neighbors(landmark_pf, left_index, right_index, trajectories_pf, semantic_landmarks_pf, semantic_landmark_index)
        list_of_used_points = np.append(list_of_used_points,direction)
        if direction == left_index:
            new_segment = sorted([int(direction), segment[1]])
        if direction == right_index:
            new_segment = sorted([segment[0], int(direction)])
    elif left_index in list_of_used_points and right_index not in list_of_used_points and right_index < trajectories_pf.shape[0]:
        list_of_used_points = np.append(list_of_used_points, right_index)
        new_segment = sorted([segment[0], int(right_index)])
    elif right_index in list_of_used_points and left_index not in list_of_used_points and left_index >= 0:
        list_of_used_points = np.append(list_of_used_points,left_index)
        new_segment = sorted([int(left_index), segment[1]])
    else:
        is_feasible = False
    return new_segment, is_feasible, list_of_used_points


"""
    This function evaluates the neighborhood of a segment using the similiraty measure 
    and returns the best direction to grow (left or right) 
    (according to RGRASP-SemTS cost function) possible.
    -----------------
    Input parameters
    -----------------
    (1) landmark_pf [float,...]
            A numpy multidimensional matrix with the point features of the landmark
    (2) left_index
            The left index value of a segment
    (3) right_index
            The right index value of a segment
    (4) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (5) semantic_landmarks_pf ([[float,...]])
            A numpy multidimensional matrix with the semantic landmarks point features
    -----------------
    Returns
    -----------------
    (1) index to grow (int)
            the decision of growing towards left, or right 
"""

def _eval_neighbors(landmark_pf, left_index, right_index, trajectories_pf, semantic_landmarks_pf, semantic_landmark_index):
    minimal_similarity,minimal_index = sys.float_info.max, sys.maxsize 
    if left_index >=0 and right_index < trajectories_pf.shape[0]:        
        # sim_left = cf._similarity(landmark_pf, trajectories_pf[left_index])
        sim_left = cf._similarity(semantic_landmarks_pf[semantic_landmark_index], trajectories_pf[left_index])
        # sim_right = cf._similarity(landmark_pf, trajectories_pf[right_index])
        sim_right = cf._similarity(semantic_landmarks_pf[semantic_landmark_index], trajectories_pf[right_index])

        if sim_left <= sim_right:
            return left_index
        else: return right_index
    
    if left_index < 0:
        return right_index   
    if right_index >= trajectories_pf.shape[0]:
        return left_index

    
"""
    This function returns the cost of all trajectory points being chosen as landmarks. 
    -----------------
    Input parameters
    -----------------
    (1) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (2) semantic_landmarks_pf ([[float,...]])
            A numpy multidimensional matrix with the semantic landmarks point features
    -----------------
    Returns
    -----------------
    (1) cost_of_candidates [[index,class_index,minimal_similarity],...]
            A multidimensional array with the the traj point index, 
            the class index for the traj point, and the similarity value
"""

def _compute_candidates_cost(trajectories_pf, semantic_landmarks_pf):
    cost_of_candidates = np.array([[i, -1.,-1.] for i in range(len(trajectories_pf))])
    for i in range(len(trajectories_pf)):
        minimal_similarity = sys.float_info.max
        minimal_index = sys.maxsize
        for j in range(len(semantic_landmarks_pf)):
            cost = cf._similarity(semantic_landmarks_pf[j], trajectories_pf[i])
            if cost < minimal_similarity: 
                minimal_similarity = cost
                minimal_index = j
        cost_of_candidates[i][1],cost_of_candidates[i][2] = minimal_index, minimal_similarity
    return cost_of_candidates