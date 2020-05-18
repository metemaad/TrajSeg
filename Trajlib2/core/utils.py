import numpy as np


def haversine(p1, p2):
    try:
        lat, lon = p1
        lat2, lon2 = p2
        d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
        distance_val = 2 * np.arcsin(np.sqrt(np.abs(a))) * 6372.8 * 1000  # convert to meter
    except Exception as e:
        print(e, p1, p2)
        distance_val = 0
    return distance_val


def get_bearing_points(lat, lon, lat2, lon2):
    lat1, lat2, diff_long = map(np.radians, (lat, lat2, lon2 - lon))
    a = np.sin(diff_long) * np.cos(lat2)
    b = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    bearing_val = np.arctan2(a, b)
    bearing_val = np.degrees(bearing_val)
    bearing_val = (bearing_val + 360) % 360

    return bearing_val

def get_bearing(row_data):
    lat = row_data.lat.values
    lon = row_data.lon.values
    lat2 = np.append(lat[1:], lat[-1:])
    lon2 = np.append(lon[1:], lon[-1:])

    lat1, lat2, diff_long = map(np.radians, (lat, lat2, lon2 - lon))
    a = np.sin(diff_long) * np.cos(lat2)
    b = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    bearing_val = np.arctan2(a, b)
    bearing_val = np.degrees(bearing_val)
    bearing_val = (bearing_val + 360) % 360
    row_data = row_data.assign(bearing=bearing_val)
    return bearing_val, row_data


def get_distance(row_data):
    lat = row_data.lat.values
    lon = row_data.lon.values
    lat2 = np.append(lat[1:], lat[-1:])
    lon2 = np.append(lon[1:], lon[-1:])
    # R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
    r = 6372.8
    d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    distance_val = 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter
    # this is the distance difference between two points not the Total distance traveled

    row_data = row_data.assign(distance=distance_val)

    return distance_val, row_data


def calculate_two_point_distance(lat, lon, lat2, lon2):
    r = 6372.8
    d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter


def distance_array(lat, lon):
    lat2 = np.append(lat[1:], lat[-1:])
    lon2 = np.append(lon[1:], lon[-1:])
    # R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
    r = 6372.8
    d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter


class StayPoint:
    latitude = None
    longitude = None
    arrive_time = None
    leave_time = None
    i = None
    j = None

    def prn(self):
        return "[" + str(self.i) + "," + str(self.j) + "]" + str(self.latitude) + "," + str(self.longitude) + ' ' + str(
            self.arrive_time) + ' ' + str(self.leave_time)

    def __init__(self):
        pass
