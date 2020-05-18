import numpy as np

from Trajlib2.core.utils import haversine


def kinematic(octal_window,verbose=False):
    #print("kin")
    if len(octal_window)!=7:
        raise Exception("kin only gets window size 7")
    lat = octal_window.lat.values
    lon = octal_window.lon.values

    #print(list(zip(lat,lon)))
    #start = octal_window.index.values.min()
    #td = (octal_window.index.values - start)  # / np.timedelta64(1, 's')
    #td=td.seconds
    td=((octal_window.index.values - octal_window.index.values.min())/1000000000).astype(float)
    #print(td)
    lat2 = np.append(lat[1:], lat[-1:])
    lon2 = np.append(lon[1:], lon[-1:])
    # R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
    # r = 6372.8
    d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
    #print(d_lat, d_lon, lat1, lat2)
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    # distance_val = 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter
    # distance_val = distance_val[:-1]
    S_lat = np.diff(lat) / td[1:]
    S_lon = np.diff(lon) / td[1:]
    a_lat = np.diff(S_lat) / td[2:]
    a_lon = np.diff(S_lon) / td[2:]
    t3 = (td[2] + td[4]) / 2
    # (t3, td[3])
    lat3 = a_lat[0] * (t3 * t3) + S_lat[1] * t3
    lon3 = a_lon[0] * (t3 * t3) + S_lon[1] * t3
    p1 = (lat[2] + lat3, lon[2] + lon3)

    lat = lat[::-1]
    lon = lon[::-1]
    td = td[::-1]

    lat2 = np.append(lat[1:], lat[-1:])
    lon2 = np.append(lon[1:], lon[-1:])
    # r = 6372.8
    d_lat, d_lon, lat1, lat2 = map(np.radians, (lat2 - lat, lon2 - lon, lat, lat2))
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    # distance_val = 2 * np.arcsin(np.sqrt(a)) * r * 1000  # convert to meter

    # distance_val = distance_val[:-1]
    S_lat = np.diff(lat) / td[1:]
    S_lon = np.diff(lon) / td[1:]
    a_lat = np.diff(S_lat) / td[2:]
    a_lon = np.diff(S_lon) / td[2:]
    # print(td)
    t3 = (td[2] + td[4]) / 2
    # (t3, td[3])
    lat3 = a_lat[0] * (t3 * t3) + S_lat[1] * t3
    lon3 = a_lon[0] * (t3 * t3) + S_lon[1] * t3
    p2 = (lat[2] + lat3, lon[2] + lon3)
    pc = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    l = (lat[3], lon[3])
    d = haversine(pc, l)
    #if np.isnan(d):
        #print(d, p1, p2, pc, l)

    return p1, p2, pc, d
