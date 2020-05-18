from Trajlib2.SegmentationAlgorithms.GRASP_UTS.cost.MDLCost import compute_cost


def calculate_cost(segments, arg, feature_bounds):
    return compute_cost(segments, arg, feature_bounds)
