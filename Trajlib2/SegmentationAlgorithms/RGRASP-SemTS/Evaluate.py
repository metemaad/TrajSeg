import numpy as np

"""
    ----------------------
    Script Description
    ----------------------
    A python script for computing the evaluation metrics of the RGRASP-SemTS Algorithm. 
    
    When using this code, please refer to: 
    
    Junior, Amilcar Soares, Valeria Cesario Times, Chiara Renso, Stan Matwin, and Lucídio AF Cabral. 
    "A semi-supervised approach for the semantic segmentation of trajectories." 
    In 2018 19th IEEE International Conference on Mobile Data Management (MDM), pp. 145-154. IEEE, 2018.    
    
    Original published version: https://ieeexplore.ieee.org/abstract/document/8411271/
    DOI: https://doi.org/10.1109/MDM.2018.00031
    Author version: https://www.researchgate.net/publication/324841537_A_Semi-Supervised_Approach_for_the_Semantic_Segmentation_of_Trajectories
    
"""

"""
    A function to compute the average purity of given solution by RGRASP-SemTS. 
    For details on the equation, please refer to the paper available at https://doi.org/10.1109/MDM.2018.00031
    -----------------
    Input parameters
    -----------------
    (1) y_true_cls [int,...]
            An array with the true class labels of the trajectory data that was segmented.
    (2) y_pred_sid [int,...]
            An array with the predicted segments' indexes (result from the algorithm's segmentation) 
    -----------------
    Returns
    -----------------
    (1) avg [float,...]
            The individual purity of each forecasted segment
    (2) avg_purity (float)
            The average purity value for all segments
    (3) seg_len
            The number of segments evaluated by the procedure
    

"""

def segment_purity(y_true_cls, y_pred_sid):
    avg = []
    y_true_cls = np.array(y_true_cls)
    y_pred_sid = np.array(y_pred_sid)
    for ts in set(y_pred_sid):
        ma = 0
        g = y_true_cls[(np.where(y_pred_sid == ts)[0])]
        for tp in set(g):
            _ = len(np.where(g == tp)[0])
            if _ > ma:
                ma = _
        if ts != -1:
            avg.append(ma * 1.0 / len(g))
    return avg, np.mean(np.array(avg)),len(avg)

"""
    A function to compute the average coverage of given solution by RGRASP-SemTS. 
    For details on the equation, please refer to the paper available at https://doi.org/10.1109/MDM.2018.00031
    -----------------
    Input parameters
    -----------------
    (1) y_true_sid [int,...]
            An array with the true segment ids of the trajectory data that was segmented.
    (2) y_pred_sid [int,...]
            An array with the predicted segments' indexes (result from the algorithm's segmentation) 
    -----------------
    Returns
    -----------------
    (1) cov [float,...]
            The individual coverage of each forecasted segment
    (2) avg_cov (float)
            The average coverage value for all segments
    

"""

def segment_coverage(y_true_sid, y_pred_sid):
    cov = []
    y_pred_sid = np.array(y_pred_sid)
    y_true_sid = np.array(y_true_sid)
    l2=[]
    for ts in set(y_true_sid):
        mx = 0
        g = y_pred_sid[(np.where(y_true_sid == ts)[0])]
        for l in set(g):
            _ = len(np.where(g == l)[0])
            if mx <= _:
                mx = _
                l2.append(l)

        cov.append(mx * 1.0 / len(g))
    return cov, np.mean(np.array(cov))