import numpy as np
import math
import sys

"""
    ----------------------
    Script Description
    ----------------------
    A python script for computing the cost functions of the RGRASP-SemTS Algorithm. 
    
    When using this code, please refer to: 
    
    Junior, Amilcar Soares, Valeria Cesario Times, Chiara Renso, Stan Matwin, and Lucídio AF Cabral. 
    "A semi-supervised approach for the semantic segmentation of trajectories." 
    In 2018 19th IEEE International Conference on Mobile Data Management (MDM), pp. 145-154. IEEE, 2018.    
    
    Original published version: https://ieeexplore.ieee.org/abstract/document/8411271/
    DOI: https://doi.org/10.1109/MDM.2018.00031
    Author version: https://www.researchgate.net/publication/324841537_A_Semi-Supervised_Approach_for_the_Semantic_Segmentation_of_Trajectories
    

"""

"""
    This is the main cost function of RGRASP-SemTS, and it receives as parameters:
    -----------------
    Input parameters
    -----------------
    (1) max_value (float)
            The maximal value to be used in the L(H) hypothesis    
    (2) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (3) landmarks_pf [[float,...]]
            A numpy multidimensional matrix with the point features of the landmarks
    (4) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments
    (5) segments_class_list [float,...]
            The class value of each segment
    (6) semantic_landmarks_pf ([[float,...]])
            A numpy multidimensional matrix with the semantic landmarks point features
    (7) semantic_landmarks_sf[[float,...]] 
            A numpy multidimensional matrix with the semantic landmarks segment features    
    (8) segments_feats_list [(string),...] (optional)
            A list with all segment features attributes to be computed 
    -----------------
    Returns
    -----------------
    (1) total (float)
            The value of the cost function given a set of segments as input
    (2) classes_index_list ([float])
            A list with the classes of each segment 
    (3)[_uns_lh_val, _uns_ldh_val, _sup_lh_res, _sup_ldh_val]
            The detailed cost of all partial functions
    

"""

def semi_supervised_cost_function(max_value, trajectories_pf, landmarks_pf, segments, segments_class_list, semantic_landmarks_pf, semantic_landmarks_sf, segments_feats_list):
    _sup_lh_res, classes_index_list_lh = _sup_lh(semantic_landmarks_pf, landmarks_pf, segments_class_list)
    seg_sizes = _calculate_seg_sizes(segments)
    _sup_ldh_val, classes_index_list_ldh = _sup_ldh(semantic_landmarks_sf, trajectories_pf, segments, segments_class_list, seg_sizes, segments_feats_list)
    _uns_lh_val = _uns_lh(max_value,landmarks_pf)
    _uns_ldh_val = _uns_ldh(landmarks_pf, trajectories_pf, segments)

    total = _uns_lh_val + _sup_lh_res + _uns_ldh_val + _sup_ldh_val
    return total, classes_index_list_lh, [_uns_lh_val, _uns_ldh_val, _sup_lh_res, _sup_ldh_val], classes_index_list_ldh


"""
    This function is the Unsupervised L(H) cost function of RGRASP-SemTS. 
    This function compares in terms of similarity the set of consecutive landmarks
    -----------------
    Input parameters
    -----------------
    (1) max_value (float)
            The maximal value to be used in the L(H) hypothesis    
    (2) landmarks_pf [[float,...]]
            A numpy multidimensional matrix with the point features of the landmarks
    -----------------
    Returns
    -----------------
    (1) value (float)
            The value of the cost function given a set of trajectory landmarks

"""

def _uns_lh(max_value, landmarks_pf):
    total_cost = 0
    #comparing landmarks 2 by 2
    for i in range(len(landmarks_pf)-1):
        s = _similarity(landmarks_pf[i], landmarks_pf[i+1])
        total_cost += max_value - s
    return math.log2(1. + total_cost)

"""
    This function is the Supervised L(H) cost function of RGRASP-SemTS, 
    and it receives as parameters:
    -----------------
    Input parameters
    -----------------
    (1) landmarks_pf [[float,...]]
            A numpy multidimensional matrix with the point features of the landmarks
    (2) semantic_landmarks_pf [(float),...]
            A numpy multidimensional matrix with the semantic landmarks point features
    (3) segments_class_list [float,...]
            The class value of each segment
    -----------------
    Returns
    -----------------
    (1) total (float)
            The value of the cost function given a set of landmarks and 
            semantic landmarks as input
    (2) classes_index_list ([float])
            A list with the classes of each segment 
"""

def _sup_lh(semantic_landmarks_pf, landmarks_pf, segments_class_list):
    total_cost = 0
    classes_index = np.array([])
    for i in range(len(landmarks_pf)):
        cost = _similarity(semantic_landmarks_pf[int(segments_class_list[i])], landmarks_pf[i])
        classes_index = np.append(classes_index, segments_class_list[i])
        total_cost += cost
    return math.log2(1.+ total_cost), classes_index

"""
    This function is the Unsupervised L(D|H) cost function of RGRASP-SemTS.
    This is the main cost function of RGRASP-SemTS, and it receives as parameters:
    -----------------
    Input parameters
    -----------------
    (1) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (2) landmarks_pf 
            A numpy multidimensional matrix with the point features of the landmarks
    (3) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments    
    -----------------
    Returns
    -----------------
    (1) total (float)
            The value of the cost function given a set of segments as input
    
"""

def _uns_ldh(landmarks_pf, trajectories_pf, segments):
    total_cost = 0
    #compares each landmark with its segment's respective trajectories points
    for i in range(len(landmarks_pf)):
        segment_pf = trajectories_pf[segments[i][0]:segments[i][1],:] 
        total_cost += _cohesiveness(landmarks_pf[i], segment_pf)        
    return math.log2(1. + total_cost)
    
"""
    This function is the Supervised L(D|H) cost function of RGRASP-SemTS.
    This is the main cost function of RGRASP-SemTS, and it receives as parameters:
    -----------------
    Input parameters
    -----------------
    (1) semantic_landmarks_sf[(float),...] 
            A numpy multidimensional matrix with the semantic landmarks segment features    
    (2) trajectories_pf ([[float,...]]) 
            A numpy multidimensional matrix with the trajectory point features. 
            These features should range from 0. -- 1 to avoid a feature being 
            prioritized instead of others.
    (3) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments
    (4) segments_class_list [float,...]
            The class value of each segment     
    (5) segments_size ([float,...])
            The size of each segment in the solution
    (6) segments_feats_list [(string),...] (optional)
            A list with all segment features attributes to be computed 
    -----------------
    Returns
    -----------------
    (1) total (float)
            The value of the cost function given a set of segments as input
    (2) classes_index_list ([float])
            A list with the classes of each segment 
    
"""
def _sup_ldh(semantic_landmarks_sf, trajectories_pf, segments, segments_class_list, seg_sizes, segments_feats_list):
    total_cost = 0
    classes_index = np.array([])
    for i in range(len(segments)):
        minimal_similarity = sys.float_info.max
        minimal_index = sys.maxsize
        segment_sf = _calculate_all_sf(trajectories_pf[segments[i][0]:segments[i][1]+1], segments_feats_list)
        cost = _similarity(semantic_landmarks_sf[int(segments_class_list[i])], segment_sf)
        classes_index = np.append(classes_index, segments_class_list[i])

    return math.log2(1.+ total_cost), classes_index

"""
    This function calculates the size of each segment
    -----------------
    Input parameters
    -----------------
    (1) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments    
    -----------------
    Returns
    -----------------
    (1) segments_size ([float,...])
            The size of each segment in the solution
"""

def _calculate_seg_sizes(segments): 
    seg_sizes = np.array([segments[0][1] - segments[0][0]+1])
    for index in range(1, len(segments)):
        seg_sizes = np.append(seg_sizes, (segments[index][1]-segments[index][0]+1))
    return seg_sizes

"""
    This function calculates all segment features of a given segment
    -----------------
    Input parameters
    -----------------
    (1) sp_f [[float,...]]
            A numpy multidimensional matrix with the point features of a given segment 
    -----------------
    Returns
    -----------------
    (1) all_seg_feats ([float,...])
            A numpy multidimensional matrix with all segment features computed for the segment
"""

def _calculate_all_sf(s_pf, segments_feats_list):
    feat_column = s_pf[:,0]
    all_seg_feats = np.array(_get_segment_sf(feat_column, segments_feats_list))
    for index in range(1, s_pf.shape[1]):
        feat_column = s_pf[:,index]
        all_seg_feats = np.concatenate((all_seg_feats, _get_segment_sf(feat_column, segments_feats_list)), axis=0)
    return all_seg_feats

"""
    This function calculates all segment features of a given column for a given segment
    -----------------
    Input parameters
    -----------------
    (1) sp_f [[float,...]]
            A numpy multidimensional matrix with the point features of a given segment 
    -----------------
    Returns
    -----------------
    (1) seg_feats ([float,...])
            A numpy multidimensional matrix with all segment features computed for the segment
"""

def _get_segment_sf(s_pf, segments_feats_list):
    seg_feats = np.array([[]])
    if 'p5' in segments_feats_list:
        seg_feats = np.append(seg_feats, np.percentile(s_pf, 5, axis=0))            
    if 'p25' in segments_feats_list:
        seg_feats = np.append(seg_feats, np.percentile(s_pf, 25, axis=0))            
    if 'p50' in segments_feats_list:
        seg_feats = np.append(seg_feats, np.percentile(s_pf, 50, axis=0))            
    if 'p75' in segments_feats_list:
        seg_feats = np.append(seg_feats, np.percentile(s_pf, 75, axis=0))                        
    if 'p95' in segments_feats_list:
        seg_feats = np.append(seg_feats, np.percentile(s_pf, 95, axis=0))
    return seg_feats

"""
    This function calculates the cohesiveness of a segment, 
    i.e. how similar the trajectory points are from its landmark
    -----------------
    Input parameters
    -----------------
    (1) landmarks_pf [[float,...]]
            A numpy multidimensional matrix with the point features of the landmarks
    (2) segments [[start_index, end_index],...]
            A numpy multidimensional matrix with the start and end positions of the 
            consecutive trajectory segments
    
    -----------------
    Returns
    -----------------
    (1) total (float)
            The value of the segment's cohesiveness
"""
    
def _cohesiveness(landmarks_pf, segment):
    #compares a landmark with all points in a segment
    landmark_array = np.empty((len(segment), *landmarks_pf.shape), landmarks_pf.dtype)
    #   copying the landmark several times to have same dimension as segments
    np.copyto(landmark_array, landmarks_pf)
    return _similarity(landmark_array, segment)


"""
    the similarity function for comparing two trajectory points or segment features
    -----------------
    Input parameters
    -----------------
    (1) tp_1 [[float,...]]
            A numpy multidimensional matrix with the point or segment features 
    (2) tp_2 [[float,...]]
            A numpy multidimensional matrix with the point or segment features 
    -----------------
    Returns
    -----------------
    (1) total (float)
            The value of the similarity for the given two multidimensional arrays
"""
def _similarity(tp1, tp2):
    return np.linalg.norm(tp1-tp2)