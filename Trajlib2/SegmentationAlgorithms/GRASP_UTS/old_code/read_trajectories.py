from Trajectories.TrajectorySample import TrajectorySample
from Trajectories.TrajectoryPoint import TrajectoryPoint
from Trajectories.TrajectorySegment import TrajectorySegment
import os
from datetime import datetime
DELIMITTER = ';'
FEATURE_BOUNDS = {'direction_inference':[],'speed_inference_m_s':[]}#,'distance_inference_m':[]

def read_hurricanes():
    print("Beginning to read Hurricanes data")
    samples = []
    feature_bounds = {'direction_inference':[],'speed_inference_m_s':[],'wind':[]}
    proper_segmentation = {}
    for filename in os.listdir(os.path.join(os.path.dirname(__file__),"databases", 'Hurricanes')):
        fname = os.path.join(os.path.dirname(__file__), "databases", 'Hurricanes', filename)
        if "maxdistances" in fname:
            continue
        file = open(fname, "r")
        content = file.readlines()
        first_line = content[0]
        headers = first_line.split(DELIMITTER)
        del content[0]
        t_sample = TrajectorySample()
        last_tid = -1
        last_sid = -1
        segments = []
        points = {}
        for line in content:
            t_point = TrajectoryPoint()
            cols = line.split(DELIMITTER)
            if last_sid != cols[2] and last_sid != -1:
                seg = TrajectorySegment()
                seg.segmentId = last_sid
                seg.points = points
                segments.append(seg)
                points = {}
            if last_tid != cols[0] and last_tid != -1:
                proper_segmentation[last_tid] = segments
                segments = []
                t_sample.tid = last_tid
                sorted_keys = sorted(t_sample.samplePoints.keys())
                t_sample.firstGid = sorted_keys[0]
                t_sample.lastGid = sorted_keys[len(sorted_keys) - 1]
                samples.append(t_sample)
                t_sample = TrajectorySample()
            t_point.tid = int(cols[0])
            last_tid = cols[0]
            t_point.gid = int(cols[1])


            t_point.latitude = float(cols[3])
            t_point.longitude = float(cols[4])
            t_point.timestamp = datetime.strptime(cols[5], "%Y-%m-%d %H:%M:%S")
            traj_features = {}
            traj_features['direction_inference'] = float(cols[6])
            feature_bounds['direction_inference'].append(float(cols[6]))
            traj_features['speed_inference_m_s'] = float(cols[7])
            feature_bounds['speed_inference_m_s'].append(float(cols[7]))
            traj_features['wind'] = float(cols[9])
            feature_bounds['wind'].append(float(cols[9]))

            t_point.trajectoryFeatures = traj_features
            t_point.label = cols[10].replace('\n', '')
            t_sample.samplePoints[t_point.gid] = t_point

            points[int(cols[1])] = t_point
            last_sid = cols[2]
        seg = TrajectorySegment()
        seg.segmentId = last_sid
        seg.points = points
        segments.append(seg)
        points = {}
        proper_segmentation[last_tid] = segments
        segments = []
        t_sample.tid = last_tid
        sorted_keys = sorted(t_sample.samplePoints.keys())
        t_sample.firstGid = sorted_keys[0]
        t_sample.lastGid = sorted_keys[len(sorted_keys)-1]
        samples.append(t_sample)
        file.close()
    return samples, proper_segmentation, feature_bounds


def read_fishing_vessels():
    print('Beginning to read Fishing Vessels')
    samples = []
    proper_segmentation = {}
    feature_bounds = {'direction_inference':[],'speed_inference_m_s':[]}

    for filename in os.listdir(os.path.join(os.path.dirname(__file__),"databases", 'Fishing Vessels')):
        fname = os.path.join(os.path.dirname(__file__), "databases", 'Fishing Vessels', filename)
        if "maxdistances" in fname:
            continue
        file = open(fname, "r")
        content = file.readlines()
        first_line = content[0]
        headers = first_line.split(DELIMITTER)
        del content[0]
        t_sample = TrajectorySample()
        last_tid = -1
        last_sid = -1
        segments = []
        points = {}
        for line in content:
            t_point = TrajectoryPoint()
            cols = line.split(DELIMITTER)
            if last_sid != cols[2] and last_sid != -1:
                seg = TrajectorySegment()
                seg.segmentId = last_sid
                seg.points = points
                segments.append(seg)
                points = {}
            if last_tid != cols[0] and last_tid != -1:
                proper_segmentation[last_tid] = segments
                segments = []
                t_sample.tid = last_tid
                sorted_keys = sorted(t_sample.samplePoints.keys())
                t_sample.firstGid = sorted_keys[0]
                t_sample.lastGid = sorted_keys[len(sorted_keys) - 1]
                samples.append(t_sample)
                t_sample = TrajectorySample()
            t_point.tid = int(cols[0])
            last_tid = cols[0]
            t_point.gid = int(cols[1])


            t_point.latitude = float(cols[3])
            t_point.longitude = float(cols[4])
            t_point.timestamp = datetime.strptime(cols[5], "%Y-%m-%d %H:%M:%S")
            traj_features = {}
            traj_features['direction_inference'] = float(cols[6])
            feature_bounds['direction_inference'].append(float(cols[6]))
            traj_features['speed_inference_m_s'] = float(cols[7])
            feature_bounds['speed_inference_m_s'].append(float(cols[7]))

            t_point.trajectoryFeatures = traj_features
            t_point.label = cols[9].replace('\n', '')
            t_sample.samplePoints[t_point.gid] = t_point

            points[int(cols[1])] = t_point
            last_sid = cols[2]
        seg = TrajectorySegment()
        seg.segmentId = last_sid
        seg.points = points
        segments.append(seg)
        points = {}
        proper_segmentation[last_tid] = segments
        segments = []
        t_sample.tid = last_tid
        sorted_keys = sorted(t_sample.samplePoints.keys())
        t_sample.firstGid = sorted_keys[0]
        t_sample.lastGid = sorted_keys[len(sorted_keys)-1]
        samples.append(t_sample)
        file.close()
    return samples, proper_segmentation, feature_bounds


def read_max_distances(directory):
    print("Beginning to read max distances")
    m = {}
    f = open(os.path.join(os.path.dirname(__file__), "databases", directory, "maxdistances.txt"))
    content = f.readlines()
    for line in content:
        cols = line.split(",")
        cols[1].replace('\n', '')
        m[cols[0]] = cols[1]
    f.close()
    return m
