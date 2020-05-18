import pandas as pd
import numpy as np


class Segment:
    trajectory_points = None
    sensitivity = 3
    speed_low_level = 0
    speed_high_level = 0
    bearing_low_level = 0
    bearing_high_level = 0
    distance_low_level = 0
    distance_high_level = 0
    speed_condition = True
    bearing_cond = True
    bearing_direction = False
    eps = None
    data_frame = None

    def get_duration_of_segment(self):
        if self.trajectory_points is None:
            print("empty segment. duration is zero.")
            return 0
        else:
            if self.trajectory_points.shape[0] < 2:
                print("only one point in segment. duration is zero")
                return 0
            else:

                start_point = self.trajectory_points[0].timeDate
                end_point = self.trajectory_points[-1].timeDate
                diff = end_point - start_point
                td = diff / np.timedelta64(1, 's')
                print("duration is " + str(td), "sec")
                return td

    def convert_to_dataframe(self):
        a = []
        for tp in self.trajectory_points:
            a = a + [[tp.timeDate, tp.latitude, tp.longitude]]
        df = pd.DataFrame(a, columns=['datetime', 'lat', 'lon'])
        df = df.set_index(['datetime'])
        df = df.sort_index()
        self.data_frame = df.copy()
        if self.trajectory_points.shape[0] > 1:
            self.calculate_features()

    def remove_first_point(self):
        if self.trajectory_points is not None:
            self.trajectory_points = self.trajectory_points[1:]
        self.convert_to_dataframe()

    def forward_window(self, tp):

        if self.trajectory_points.shape is None:
            self.add_point(tp)
        else:
            # remove first point
            self.trajectory_points = self.trajectory_points[1:]
            # add tp to the end of segment
            self.add_point(tp)

        # update calculations
        self.eps = self.calculate_features()
        print("updated eps:", self.eps)

    #        self.plot_segment()

    def add_point(self, trajectory_point):
        if self.trajectory_points is None:
            self.trajectory_points = np.array([trajectory_point])
        else:
            self.trajectory_points = np.append(self.trajectory_points, np.array([trajectory_point]))
        self.convert_to_dataframe()

    def get_length(self):
        return self.data_frame.shape[0]

    def __init__(self, trajectory_point, sensitivity, speed_condition=True, bearing_cond=True):
        self.speed_condition = speed_condition
        self.bearing_cond = bearing_cond
        # sensitivity decrease or increase the details of segments

        self.sensitivity = sensitivity
        # self.Trajectory_points = [trajectory_point]
        self.add_point(trajectory_point)
        print(self.data_frame)

    def get_duration(self):

        t = np.diff(pd.to_datetime(self.data_frame.index)) / 1000000000  # convert to second
        t = t.astype(np.float64)
        t = np.append(t[0:], t[-1:])

        tmp = self.data_frame.assign(td=t)
        tmp1 = tmp.loc[tmp['td'] > 0, :]
        # avoid NaN in case rate of sampling is more than 1 per second
        self.data_frame = tmp1.copy()

        return tmp1

    def get_distance(self):
        lat = self.data_frame.lat.values.astype(np.float64)
        lon = self.data_frame.lon.values.astype(np.float64)
        lat2 = np.append(lat[1:], lat[-1:])
        lon2 = np.append(lon[1:], lon[-1:])
        # R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
        r = 6372.8
        d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
        distance_val = 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter

        self.data_frame = self.data_frame.assign(distance=distance_val)

        return distance_val

    def get_speed(self):
        speed_val = self.data_frame['distance'] / self.data_frame['td']

        self.data_frame = self.data_frame.assign(speed=speed_val)

        return speed_val

    def get_bearing(self):
        lat = self.data_frame.lat.values.astype(np.float64)
        lon = self.data_frame.lon.values.astype(np.float64)
        lat2 = np.append(lat[1:], lat[-1:])
        lon2 = np.append(lon[1:], lon[-1:])

        lat1, lat2, diff_long = map(np.radians, (lat, lat2, lon2 - lon))
        a = np.sin(diff_long) * np.cos(lat2)
        b = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
        bearing_val = np.arctan2(a, b)
        bearing_val = np.degrees(bearing_val)
        bearing_val = (bearing_val + 360) % 360

        self.data_frame = self.data_frame.assign(bearing=bearing_val)
        return bearing_val

    def calculate_features(self):
        self.get_duration()
        self.get_distance()
        self.get_speed()
        self.get_bearing()
        # print("eps=mean(distance)+std(distance)")
        self.distance_low_level = self.data_frame['distance'].mean() - self.sensitivity * self.data_frame[
            'distance'].std()
        self.distance_high_level = self.data_frame['distance'].mean() + self.sensitivity * self.data_frame[
            'distance'].std()
        # self.speed_low_level = self.data_frame['speed'].mean() - self.sensitivity * self.data_frame['speed'].std()
        # self.speed_high_level = self.data_frame['speed'].mean() + self.sensitivity * self.data_frame['speed'].std()

        self.speed_low_level = self.data_frame['speed'].abs().mean() - self.sensitivity * self.data_frame[
            'speed'].abs().std()
        self.speed_high_level = self.data_frame['speed'].abs().mean() + self.sensitivity * self.data_frame[
            'speed'].abs().std()

        # self.speed_low_level,self.speed_high_level=self.data_frame['speed'].quantile([0.25, 0.75])

        self.bearing_low_level = (self.data_frame['bearing'].mean() - self.sensitivity * self.data_frame[
            'bearing'].std()) % 360
        self.bearing_high_level = (self.data_frame['bearing'].mean() + self.sensitivity * self.data_frame[
            'bearing'].std()) % 360
        if self.bearing_low_level > self.bearing_high_level:
            tmp = self.bearing_low_level
            self.bearing_low_level = self.bearing_high_level
            self.bearing_high_level = tmp
        if (self.data_frame['bearing'].mean() <= self.bearing_high_level) and (
                self.data_frame['bearing'].mean() >= self.bearing_low_level):
            # outside is accepted
            self.bearing_direction = True
        else:
            self.bearing_direction = False

        # =self.speed_low_level, self.speed_high_level, self.bearing_low_level, self.bearing_high_level, self.bearing_direction

        return self.speed_low_level, self.speed_high_level, self.bearing_low_level, self.bearing_high_level, self.bearing_direction

    def get_eps(self):

        return self.speed_low_level, self.speed_high_level, self.bearing_low_level, self.bearing_high_level, self.bearing_direction

    def get_all_eps(self):
        return self.speed_low_level, self.speed_high_level, self.bearing_low_level, self.bearing_high_level, self.bearing_direction

    def reset(self, trajectory_points):
        self.data_frame = None

        for tp in trajectory_points:
            self.add_point(tp)

        self.convert_to_dataframe()

    def cluster(self, prev_tp):
        """
        Check if the first point of the segment belongs to the cluster

        """
        # first point belongs to first cluster
        if prev_tp is None:
            return True

        eps = self.eps

        curr_tp = self.trajectory_points[0]

        s = np.abs(prev_tp.speed(curr_tp))
        b = prev_tp.bearing(curr_tp)
        if (eps[0] <= s) and (s < eps[1]):#and ((eps[2] <= b < eps[3]) == eps[4]):
            print(" inside")
            return True

        else:
            print("outside")
            return False
