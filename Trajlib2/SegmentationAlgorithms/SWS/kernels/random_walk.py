import numpy as np
from Trajlib2.core.utils import get_bearing, get_distance, haversine


def random_walk2(octal_window, verbose=False):
    # print("running random walk 2")
    d = get_distance(octal_window)[0][:-1]
    l = d.mean()
    ls = d.std()
    b = get_bearing(octal_window)[0][:-1]
    t = b.mean()
    ts = b.std()

    l = np.random.normal(l, ls, 1)
    t = np.radians(np.random.normal(t, ts, 1))
    pl = octal_window.iloc[3, :]

    p1 = octal_window.iloc[2, :]
    p2 = octal_window.iloc[4, :]
    dy = l * np.cos(t)
    dx = l * np.sin(t)
    r_earth = 6378000
    pi = np.pi
    new_latitude = p1.lat + (dy / r_earth) * (180 / pi)
    new_longitude = p1.lon + (dx / r_earth) * (180 / pi) / np.cos(p1.lat * pi / 180)

    pc = (new_latitude, new_longitude)
    d = float(haversine(pc, (pl.lat, pl.lon)))

    return p1, p2, pc, d


def random_walk(octal_window):

    if len(octal_window) < 3:
        raise Exception("window size should be more than 3")

    mid = int(len(octal_window) / 2)
    pl = octal_window.iloc[mid, :]
    reverse_octal_windows = octal_window[::-1]

    d = get_distance(octal_window)[0][:mid + 1]
    l = d.mean()
    ls = d.std()

    b = get_bearing(octal_window)[0][:mid + 1]
    t = b.mean()
    ts = b.std()

    l = np.random.normal(l, ls, 1)
    t = np.radians(np.random.normal(t, ts, 1))

    p1 = octal_window.iloc[mid - 1, :]

    dy = l * np.cos(t)
    dx = l * np.sin(t)
    r_earth = 6378000
    pi = np.pi
    new_latitude = p1.lat + (dy / r_earth) * (180 / pi)
    new_longitude = p1.lon + (dx / r_earth) * (180 / pi) / np.cos(p1.lat * pi / 180)

    pf = (new_latitude, new_longitude)

    d = get_distance(reverse_octal_windows)[0][:mid + 1]
    l = d.mean()
    ls = d.std()

    b = get_bearing(reverse_octal_windows)[0][:mid + 1]
    t = b.mean()
    ts = b.std()

    l = np.random.normal(l, ls, 1)
    t = np.radians(np.random.normal(t, ts, 1))

    p2 = reverse_octal_windows.iloc[mid - 1, :]

    dy = l * np.cos(t)
    dx = l * np.sin(t)
    r_earth = 6378000
    pi = np.pi
    new_latitude = p2.lat + (dy / r_earth) * (180 / pi)
    new_longitude = p2.lon + (dx / r_earth) * (180 / pi) / np.cos(p2.lat * pi / 180)

    pb = (new_latitude, new_longitude)

    pc = ((pf[0] + pb[0]) / 2, (pf[1] + pb[1]) / 2)

    d = float(haversine(pc, (pl.lat, pl.lon)))

    return p1, p2, pc, d
