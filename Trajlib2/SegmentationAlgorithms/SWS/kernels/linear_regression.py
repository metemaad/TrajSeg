from Trajlib2.core.utils import haversine
import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression(octal_window, verbose=False):
    lat = np.array(octal_window.lat.values)
    lon = np.array(octal_window.lon.values)
    td = ((octal_window.index.values - octal_window.index.values.min()) / 1000000000).astype(float)
    mid = int(len(lon) / 2)

    lat_f = lat[:mid]
    td_f = td[:mid]
    # print(lat_f, td[mid])
    LR_lat = LinearRegression()
    LR_lat.fit(td_f.reshape(-1, 1), lat_f.reshape(-1, 1))
    y_pred_lat = LR_lat.predict(np.array(td[mid]).reshape(-1, 1))
    # print(y_pred_lat[0][0])
    lat_f = y_pred_lat[0][0]
    # print(LR_lat.intercept_, LR_lat.coef_)
    # abline(LR_lat.coef_[0], LR_lat.intercept_[0])
    # plt.scatter(td[mid], y_pred_lat[0][0], c='r')

    lat_b = lat[mid:][::-1]
    td_b = td[mid:][::-1]
    # print(lat_b, td[mid])
    LR_lat = LinearRegression()
    LR_lat.fit(td_b.reshape(-1, 1), lat_b.reshape(-1, 1))
    y_pred_lat = LR_lat.predict(np.array(td[mid]).reshape(-1, 1))
    # print(y_pred_lat[0][0])

    # print(LR_lat.intercept_, LR_lat.coef_)
    # abline(LR_lat.coef_[0], LR_lat.intercept_[0])

    # plt.scatter(td[mid], y_pred_lat[0][0], c='r')
    lat_b = y_pred_lat[0][0]
    lat_pred = (lat_f + lat_b) / 2
    # plt.scatter(td[mid], lat_pred, c='g')
    # plt.show()

    # print("lon")
    # plt.scatter(td, lon)
    mid = int(len(lon) / 2)
    # td =np.array(((octal_window.index.values - octal_window.index.values.min()) / 1000000000).astype(float))
    lon_f = lon[:mid]
    td_f = td[:mid]
    # print(lon_f, td[mid])
    LR_lon = LinearRegression()
    LR_lon.fit(td_f.reshape(-1, 1), lon_f.reshape(-1, 1))
    y_pred_lon = LR_lon.predict(np.array(td[mid]).reshape(-1, 1))
    # print(y_pred_lon[0][0])
    lon_f = y_pred_lon[0][0]
    # print(LR_lon.intercept_, LR_lon.coef_)
    # abline(LR_lon.coef_[0], LR_lon.intercept_[0])
    # plt.scatter(td[mid], y_pred_lon[0][0], c='y')

    lon_b = lon[mid:][::-1]
    td_b = td[mid:][::-1]
    # print(lon_b, td[mid])
    LR_lon = LinearRegression()
    LR_lon.fit(td_b.reshape(-1, 1), lon_b.reshape(-1, 1))
    y_pred_lon = LR_lon.predict(np.array(td[mid]).reshape(-1, 1))
    # print(y_pred_lon[0][0])

    # print(LR_lon.intercept_, LR_lon.coef_)
    # abline(LR_lon.coef_[0], LR_lon.intercept_[0])
    # plt.scatter(td[mid], y_pred_lon[0][0], c='r')
    lon_b = y_pred_lon[0][0]
    lon_pred = (lon_f + lon_b) / 2
    # plt.scatter(td[mid], lon_pred, c='g')
    # plt.show()
    # print(lat[mid], lon[mid])
    # print(lat_pred, lon_pred)
    pc=(lat[mid], lon[mid])
    p1=p2=(lat_pred, lon_pred)
    d = haversine((lat[mid], lon[mid]), (lat_pred, lon_pred))
    return p1, p2, pc, d
    # print(d)
