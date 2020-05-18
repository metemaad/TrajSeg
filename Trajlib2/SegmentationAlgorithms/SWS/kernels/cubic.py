import numpy as np
from Trajlib2.core.utils import  haversine
from scipy.interpolate import interp1d


def cubic(octal_window,verbose=False):
    if len(octal_window)<3:
        raise Exception("window size should be more than 3")
    try:
        lat = octal_window.lat.values
        lon = octal_window.lon.values
        mid = int(len(lon) / 2)

        x3 = lat[mid]
        y3 = lon[mid]
        lat = np.delete(lat, mid)
        lon = np.delete(lon, mid)
        t = np.diff(octal_window.index) / 1000000000
        t3 = t[mid]
        t = np.delete(t, mid)
        t = np.cumsum(t)
        t = np.insert(t, 0, 0).astype(float)

        from scipy.interpolate import CubicSpline

        latcs = CubicSpline(np.abs(t), lat)

        new_x = latcs(t3)

        loncs = CubicSpline(np.abs(t), lon)
        new_y = loncs(t3)

        #fx = interp1d(t, lat, kind='cubic')
        #fy = interp1d(t, lon, kind='cubic')
        #new_x = fx(t3)
        #new_y = fy(t3)

        pf = (new_x, new_y)




        #reverse
        lat = octal_window.lat.values[::-1]
        lon = octal_window.lon.values[::-1]
        tidx=octal_window.index[::-1]



        x3 = lat[mid]
        y3 = lon[mid]
        lat = np.delete(lat, mid)
        lon = np.delete(lon, mid)
        t = np.diff(tidx) / 1000000000
        t3 = t[mid]
        t = np.delete(t, mid)
        t = np.cumsum(t)
        t = np.insert(t, 0, 0).astype(float)

        from scipy.interpolate import CubicSpline

        latcs = CubicSpline(np.abs(t), lat)

        new_x =latcs(t3)

        loncs = CubicSpline(np.abs(t), lon)
        new_y =loncs(t3)

        #fx = interp1d(t, lat, kind='cubic',fill_value="extrapolate")

        #fy = interp1d(t, lon, kind='cubic',fill_value="extrapolate")
        #new_x = fx(t3)
        #new_y = fy(t3)
        pb = (new_x, new_y)
        pc = ((pf[0] + pb[0]) / 2, (pf[1] + pb[1]) / 2)



        d = haversine(pc, (x3, y3))

    except Exception as e:
        if verbose:
            print(t,e)
            print(octal_window)
        d=0
    return pf, pb, pc, d