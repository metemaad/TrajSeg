from Trajlib2.core.utils import haversine


def linear(octal_window,verbose=False):
    #if len(octal_window)!=5:
    #    raise Exception("Linear only get window size 3")
    mid = int(len(octal_window) / 2)
    lat = octal_window.lat.values
    lon = octal_window.lon.values
    pc = ((lat[mid-1] + lat[mid+1]) / 2, (lon[mid-1] + lon[mid+1]) / 2)
    d = haversine(pc, (lat[mid], lon[mid]))
    return (lat[mid-1], lon[mid-1]), (lat[mid+1], lon[mid+1]), pc, d
