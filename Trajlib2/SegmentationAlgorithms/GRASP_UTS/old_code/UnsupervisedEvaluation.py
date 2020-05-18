def calculate_purity(segment):
    labels = {}
    for gid in segment.points:
        label = segment.points[gid].label
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1
    max_labels = 0
    for key in labels:
        if labels[key] > max_labels:
            max_labels = labels[key]
    return max_labels / len(segment.points)


def calculate_coverage(segment, proper_segmentation):
    max_coverage = 0.0
    for s in proper_segmentation:
        s.compute_segment_features()
        coverage = 0.0
        for gid in segment.points:
            if gid >= s.firstGid and gid <= s.lastGid:
                coverage+=1

        coverage /= len(s.points)
        if coverage > max_coverage:
            max_coverage = coverage

    return max_coverage