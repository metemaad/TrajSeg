import pandas as pd
import math
import numpy as np


class TrajectoryPoint:
    latitude = 0
    longitude = 0
    timeDate = 0
    label=None
    transportation_mode=None
    trajectory_id=None

    def duration(self, tp):
        diff = self.timeDate - tp.timeDate
        td = diff / np.timedelta64(1, 's')

        return np.abs(td)

    def distance(self, tp):
        # print("td:",str(self.duration(tp)))
        r = 6372.8
        d_lat, d_lon, lat1, lat2 = map(math.radians, (
            tp.latitude - self.latitude, tp.longitude - self.longitude, self.latitude, tp.latitude))
        a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
        d = 2 * np.arcsin(math.sqrt(a)) * r * 1000
        #print("distance ", str(self.latitude), ",", str(self.longitude), " from", str(tp.latitude), ",",
        #      str(tp.longitude), "  =", str(d))
        return d

    def speed(self, tp):
        td = self.duration(tp)
        if td > 0:
            d = self.distance(tp)
            s = d / td

            print("speed ", str(self.latitude), ",", str(self.longitude), " from", str(tp.latitude), ",",
                  str(tp.longitude), "  =", str(s),"|",str(d),str(td))
            return s
        else:
            print("speed ", str(self.latitude), ",", str(self.longitude), " from", str(tp.latitude), ",",
                  str(tp.longitude), "  =0","|",str(td))
            return 0

    def __init__(self, lat, lon, time_date,label=None,mode=None,trajectory_id=None):
        self.latitude = float(lat)
        self.longitude = float(lon)
        self.timeDate = pd.to_datetime(time_date)
        self.label=label
        self.transportation_mode=mode
        self.trajectory_id=trajectory_id
        # print("new Trajectory point:", lat, lon, time_date)

    def bearing(self, tp):
        """

        :type tp: TrajectoryPoint
        """
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(tp.latitude)
        diff = math.radians(self.longitude - tp.longitude)
        x = math.sin(diff) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff))
        b = (math.degrees(math.atan2(x, y)) + 360) % 360
        #print("b:",b)
        return b














