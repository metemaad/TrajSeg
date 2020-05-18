def maximum_quantity_of_segments(traj, min_time):

    d1 = traj.samplePoints[traj.lastGid].timestamp
    d2 = traj.samplePoints[traj.firstGid].timestamp
    interval = d1 - d2
    time_elapsed = (interval.days * 86400) + interval.seconds
    quantity_of_points = len(traj.samplePoints)
    average_sampling_time = round(time_elapsed / quantity_of_points)
    quantity_of_points_per_segment = 0
    time = 0
    if average_sampling_time <= 0:
        average_sampling_time = 1
    while time <= min_time:
        time += average_sampling_time
        quantity_of_points_per_segment += 1
    maximum_quantity_of_segments = 0
    qnt_of_points = 0
    while qnt_of_points <= quantity_of_points:
        qnt_of_points += quantity_of_points_per_segment
        maximum_quantity_of_segments += 1
    return maximum_quantity_of_segments