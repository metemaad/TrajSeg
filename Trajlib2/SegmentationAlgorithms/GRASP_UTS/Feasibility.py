from datetime import timedelta


def is_feasible(segments, min_time):
    if len(segments)<1:
        return False
    for s in segments:
        keys = list(s.points.keys())
        keys.sort()
        time_elapsed = -1
        if len(keys)>1:
            d1 = s.points[keys[len(keys) - 1]].timestamp
            d2 = s.points[keys[0]].timestamp
            time_delta = timedelta()
            time_delta = d1-d2
            time_elapsed = (time_delta.days * 86400) + time_delta.seconds
        if time_elapsed < min_time:
            return False
    return True