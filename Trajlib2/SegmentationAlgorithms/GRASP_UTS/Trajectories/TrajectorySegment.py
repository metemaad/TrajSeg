from Trajlib2.SegmentationAlgorithms.GRASP_UTS.Trajectories.TrajectoryPoint import TrajectoryPoint
import math


class TrajectorySegment:

    def __init__(self):
        self.segmentId = 0
        self.label = ""
        self.landmark = TrajectoryPoint()
        self.points = {}  # indexed by gid
        self.segmentFeatures = {}  # name, value
        self.firstGid = 0
        self.lastGid = 0

    def compute_segment_features(self):
        feats = {}
        keys = list(self.points.keys())
        list.sort(keys)
        self.firstGid = keys[0]
        self.lastGid = keys[len(keys) - 1]
        #feats = self.run_general_segment_feats()
        #self.segmentFeatures = feats



    def calculate_distance_meters(self, p1, p2):
        R = 6371  # km

        lat1 = p1.latitude
        lat2 = p2.latitude
        lon1 = p1.longitude
        lon2 = p2.longitude
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat / 2) * math.sin(dLat / 2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dLon / 2) * math.sin(dLon / 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c
        return d * 1000  # meters

    def copy(self):
        s = TrajectorySegment()
        s.segmentId = self.segmentId
        s.landmark = self.landmark.copy()
        #print(s.label)
        s.label = self.label
        for key in self.points.keys():
            s.points[key] = self.points[key].copy()
        for key in self.segmentFeatures.keys():
            s.segmentFeatures[key] = self.segmentFeatures[key]
        s.firstGid = self.firstGid
        s.lastGid = self.lastGid
        return s
