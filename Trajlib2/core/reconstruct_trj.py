#!/usr/bin/env python
# coding: utf-8

# In[39]:



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
def get_bearing2(lat,lon,lat2,lon2):
    
    lat1, lat2, diff_long = map(np.radians, (lat, lat2, lon2 - lon))
    a = np.sin(diff_long) * np.cos(lat2)
    b = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    bearing_val = np.arctan2(a, b)
    bearing_val = np.degrees(bearing_val)
    bearing_val = (bearing_val + 360) % 360
    
    return bearing_val

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





def random_walk2(octal_window,verbose=False):
    #print("running random walk 2")
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
    new_latitude = p1.lat + (dy / r_earth) * (180 / pi);
    new_longitude = p1.lon + (dx / r_earth) * (180 / pi) / np.cos(p1.lat * pi / 180);

    pc = (new_latitude, new_longitude)
    d = float(haversine(pc, (pl.lat, pl.lon)))

    return p1, p2, pc, d
def random_walk(octal_window):
    #print(len(octal_window))
    if len(octal_window)<3:
        raise Exception("window size should be more than 3")

    mid=int(len(octal_window)/2)
    pl = octal_window.iloc[mid, :]
    reverse_octal_windows=octal_window[::-1]

    d = get_distance(octal_window)[0][:mid+1]
    l = d.mean()
    ls = d.std()

    b = get_bearing(octal_window)[0][:mid+1]
    t = b.mean()
    ts = b.std()

    l = np.random.normal(l, ls, 1)
    t = np.radians(np.random.normal(t, ts, 1))


    p1 = octal_window.iloc[mid-1, :]

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
    new_latitude = p2.lat + (dy / r_earth) * (180 / pi);
    new_longitude = p2.lon + (dx / r_earth) * (180 / pi) / np.cos(p1.lat * pi / 180);

    pb = (new_latitude, new_longitude)

    pc=((pf[0]+pb[0])/2,(pf[1]+pb[1])/2)

    d = float(haversine(pc, (pl.lat, pl.lon)))




    return p1, p2, pc, d


# In[184]:


lat=0
lon=0
lat2=10
lon2=10
time=50
t=20
ts=5
l=100 #fixed l=dist/time
ls=l/10 #=0

def func(start,l,t,ts):
    (lat,lon)=start
    #l = np.random.normal(l, ls, 1) #no need
    t = np.radians(np.random.normal(t, ts, 1))
    dy = l * np.cos(t)
    dx = l * np.sin(t)
    r_earth = 6378000
    pi = np.pi
    new_latitude = lat + (dy / r_earth) * (180 / pi);
    new_longitude = lon + (dx / r_earth) * (180 / pi) / np.cos(lat * pi / 180);
    return new_latitude,new_longitude



# In[185]:


func((0,0),1,10,2)


# In[382]:



end=(1,1,0,0)
start=(0,0,0,0)
def reconst(start=(0,0,0,0,0),end=(1,1,0,0,0),density=0.01,s=0.1,
            plot=False,time_threshold=2000):

    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    x,y,ts,l,st=start
    p,q,tsp,lp,et=end
    #print(st,et)
    if et==st:
        return None
    delta=st-et
    if np.abs(delta.seconds)>time_threshold:
        res=[[x,y,ts,l,st],[p,q,tsp,lp,et]]
        return pd.DataFrame(res,columns=['lat','lon','tsid','label','time'])
        
    
    n=int(delta.seconds*density)
    ptd=pd.Timedelta(((et-st).seconds/n)*1000000000)  
    
    for i in range(n-1):

        e.append(st+ptd*(i+1))


        ll=calculate_two_point_distance(start[0],start[1],end[0],end[1])-calculate_two_point_distance(start[0],start[1],x,y)
        br=get_bearing2(x,y,end[0],end[1])
        x,y=func((x,y),ll/(n-i), br,br*s)
        #print(x[0],y[0],a,b)
        
        
        a.append(x[0].astype(float))
        b.append(y[0].astype(float))
        if tsp==ts:
            c.append(ts)
        else:
            if i<int(n/2):
                c.append(ts)
            else:
                c.append(tsp)
        if lp==l:
            d.append(l)
        else:
            if i<int(n/2):
                d.append(l)
            else:
                d.append(lp)
        #print(x,y,ll,br)
    if plot:
        from matplotlib import pyplot as plt
        plt.scatter(a,b,color='b',s=5)
        plt.scatter(start[0],start[1],color='r')
        plt.scatter(end[0],end[1],color='g')
    re=list(zip(a,b,c,d,e))
    return pd.DataFrame(re,columns=['lat','lon','tsid','label','time'])

#x=13
##start=(lon[x],lat[x],tsid[x],label[x],timea[x])
#end=(lon[x+1],lat[x+1],tsid[x+1],label[x+1],timea[x+1])

#n=int(datediff[x]/500)
#res=reconst(start,end,density=0.0001,s=0.02,plot=True)
#print(res)
#x=14
#start=(lon[x],lat[x],tsid[x],label[x],timea[x])
#end=(lon[x+1],lat[x+1],tsid[x+1],label[x+1],timea[x+1])

#n=int(datediff[x]/500)
#res2=reconst(start,end,density=0.0001,s=0.02,plot=True)

#plt.show()


# In[383]:


df=pd.DataFrame()
df=df.append(res.append(res2))

df=df.reset_index()
df=df.drop(columns=['index'])
df


# In[384]:


delta=timea[1]-timea[0]
delta
b=pd.Timedelta(60*1000000000)
timea[1]+b


# In[ ]:





# In[385]:



datediff=np.diff(pd.to_datetime(df.time))/1000000000
plt.plot(datediff.astype(float))
plt.ylim([0,10000])


# In[390]:


import pandas as pd
def reconstruct_traj(filename="/Users/mohammadetemad/trajectory-segmentation/trajlib_v2/databases/fishing/fv_d3.txt",
                     sep=';',density=0.001,std=0.002,time_threshold=2000):
    df=pd.read_csv(filename,sep=sep)
    df=df.assign(dt=pd.to_datetime(df.time))
    df=df.set_index(['tid','dt'])
    df=df.sort_index()
    df=df.reset_index()
    #time_threshold=2000
    tsid=df.tid*100000+df.sid
    label=df.label
    timea=pd.to_datetime(df.time)

    datediff=np.diff(pd.to_datetime(df.time))/1000000000
    lat=df.latitude
    lon=df.longitude
    df=pd.DataFrame()



    for x in range(len(datediff)):

        start=(lon[x],lat[x],tsid[x],label[x],timea[x])
        end=(lon[x+1],lat[x+1],tsid[x+1],label[x+1],timea[x+1])
        if start==end:
            continue



        res2=reconst(start,end,density=density,s=std,plot=True,time_threshold=time_threshold)
        df=df.append(res2)

        df=df.reset_index()
        df=df.drop(columns=['index'])
    #print(df)
    return df

def plot_df(df):
    color=['g','b','k','pink','yellow']
    i=0
    for k,v in df.groupby(['label']):
        i=(i+1)%2
        #print(i)
        plt.scatter(v.lat,v.lon,s=100,color=color[i],alpha=.4)
    color=['orange','red','k','pink','yellow']
    i=0
    for k,v in df.groupby(['tsid']):
        i=(i+1)%5
        #print(i)
        plt.scatter(v.lat,v.lon,s=5,color=color[i],alpha=.6)
    #plt.xlim([-35.15,-34.19])
    #plt.ylim([0,1.3])
    plt.show()
df=reconstruct_traj(density=0.0003,time_threshold=2000)
plot_df(df)
print(df.shape)
df=pd.read_csv("/Users/mohammadetemad/trajectory-segmentation/trajlib_v2/databases/fishing/fv_d3.txt",
                     sep=';')
#print(df.columns)
df.columns=['tid', 'gid', 'sid', 'lon', 'lat', 'time',
       'direction_inference', 'speed_inference_m_s', 'distance_inference_m',
       'label']
df=df.assign(tsid=df.tid*100000+df.sid)
plot_df(df)
print(df.shape)


# In[ ]:


df=pd.read_csv("/Users/mohammadetemad/trajectory-segmentation/trajlib_v2/databases/fishing/fv_d3.txt",
                     sep=';')


# In[ ]:


df=df.assign(dt=pd.to_datetime(df.time))
df=df.set_index(['dt'])
df=df.sort_index()
df=df.reset_index()
df


# In[ ]:





# In[ ]:




