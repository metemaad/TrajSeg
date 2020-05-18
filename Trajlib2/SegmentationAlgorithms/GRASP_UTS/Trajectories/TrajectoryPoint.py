from datetime import datetime
import copy,numpy as np

class TrajectoryPoint:
    def __init__(self):
        self.gid = 0
        self.latitude = 0.0
        self.longitude = 0.0
        self.timestamp = datetime.now()  # This needs to be a timestamp
        self.tid = 0
        self.trajectoryFeatures = {}  # name, value
        self.label = ""
        self.point_vector = np.array([])

    def copy(self):
        t = TrajectoryPoint()
        t.gid = self.gid
        t.latitude = self.latitude
        t.longitude = self.longitude
        t.timestamp = self.timestamp
        t.tid = self.tid
        t.trajectoryFeatures = self.trajectoryFeatures
        t.label = self.label
        t.point_vector = self.point_vector
        return t
