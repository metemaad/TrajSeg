def segment_by_label(df, label='transportation_mode'):

    t = df[label].values
    start = 0
    segments = []
    for i in range(len(t) - 1):
        if t[i] != t[i + 1]:
            segments.append([start, i + 1])
            start = i + 1
    segments.append([start, len(t)])
    trajectory_segments = {}
    i = 0
    for segment in segments:
        trajectory_segments[i] = df.iloc[segment[0]:segment[1], :]
        i = i + 1
    del df
    del t
    return segments, trajectory_segments