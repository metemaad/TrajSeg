"""

# Best Track Data (HURDAT2)

Atlantic hurricane database (HURDAT2) 1851-2018 (5.9MB download)
This dataset was provided on 10 May 2019 to include the 2018 update to the best tracks.

This dataset (known as Atlantic HURDAT2) has a comma-delimited, text format with six-hourly information on the location, maximum winds, central pressure, and (beginning in 2004) size of all known tropical cyclones and subtropical cyclones. The original HURDAT database has been retired.

Detailed information regarding the Atlantic Hurricane Database Re-analysis Project is available from the Hurricane Research Division.

ref:https://www.nhc.noaa.gov/data/

#https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atlantic.pdf
# HURDAT2 Processor

This is a python script that convert your HURDAT to a dataframe and generate a CSV file for you to eaily process this data. This work is part of trajectory segmentation research[1]. We use this dataset for evaluation purposes. If you are going to apply this script please cite to our work.
Thanks.
[1]: Etemad, Mohammad, et al. "A Trajectory Segmentation Algorithm Based on Interpolation-based Change Detection Strategies." EDBT/ICDT Workshops. 2019.

"""
import pandas as pd
Record_identifier_dic={'C':"Closest approach to a coast, not followed by a landfall"
,'G':"Genesis"
,'I':"An intensity peak in terms of both pressure and wind"
,'L':"Landfall (center of system crossing a coastline)"
,'P':"Minimum in central pressure"
,'R':"Provides additional detail on the intensity of the cyclone when rapid changes are underway"
,'S':"Change of status of the system"
,'T':"Provides additional detail on the track (position) of the cyclone"
,'W':"Maximum sustained wind speed"}
Status_of_system_dic={
'TD':"Tropical cyclone of tropical depression intensity (< 34 knots)"
,'TS':"Tropical cyclone of tropical storm intensity (34-63 knots)"
,'HU':"Tropical cyclone of hurricane intensity (> 64 knots)"
,'EX':"Extratropical cyclone (of any intensity)"
,'SD':"Subtropical cyclone of subtropical depression intensity (< 34 knots)"
,'SS':"Subtropical cyclone of subtropical storm intensity (> 34 knots)"
,'LO':"A low that is neither a tropical cyclone, a subtropical cyclone, nor an extratropical cyclone (of any intensity)"
,'WV':"Tropical Wave (of any intensity)"
,'DB':"Disturbance (of any intensity)"} 
    
def process_details(data):
    data=data.split(',')
    Year=int(data[0][0:4])
    Month=int(data[0][4:6])
    Day=int(data[0][6:8])
    Hours_in_UTC=int(data[1][0:2])
    Minutes_in_UTC=int(data[1][2:4])
    Record_identifier=data[2].strip()
    try:
        Record_identifier_desc=Record_identifier_dic[data[2].strip()]
    except:
        Record_identifier_desc=None
        
    Status_of_system=data[3].strip()
    try:
        Status_of_system_desc=Status_of_system_dic[Status_of_system]
    except:
        Status_of_system_desc=None
        
    if data[4].strip()[-1:] in ('N','S'):
        if data[4].strip()[-1:]=='N':
            Latitude=float(data[4].strip()[:-1])
        else:
            Latitude=-1.0*float(data[4].strip()[:-1])
    else:
        Latitude=-999
    
    if data[5].strip()[-1:] in ('E','W'):
        if data[5].strip()[-1:]=='E':
            Longitude=float(data[5].strip()[:-1])
        else:
            Longitude=-1.0*float(data[5].strip()[:-1])
    else:
        Longitude=-999
    Maximum_sustained_wind_in_knots=float(data[6].strip())
    Minimum_Pressure_in_millibars=float(data[7].strip())
    i=8
    F34_kt_wind_radii_maximum_northeastern=float(data[i].strip())
    i+=1
    F34_kt_wind_radii_maximum_southeastern=float(data[i].strip())
    i+=1
    F34_kt_wind_radii_maximum_southwestern=float(data[i].strip())
    i+=1
    F34_kt_wind_radii_maximum_northwestern=float(data[i].strip())
    

    i+=1
    F50_kt_wind_radii_maximum_northeastern=float(data[i].strip())
    i+=1
    F50_kt_wind_radii_maximum_southeastern=float(data[i].strip())
    i+=1
    F50_kt_wind_radii_maximum_southwestern=float(data[i].strip())
    i+=1
    F50_kt_wind_radii_maximum_northwestern=float(data[i].strip())
    
    i+=1
    F64_kt_wind_radii_maximum_northeastern=float(data[i].strip())
    i+=1
    F64_kt_wind_radii_maximum_southeastern=float(data[i].strip())
    i+=1
    F64_kt_wind_radii_maximum_southwestern=float(data[i].strip())
    i+=1
    F64_kt_wind_radii_maximum_northwestern=float(data[i].strip())


    
    res=Year,Month,Day,Hours_in_UTC,Minutes_in_UTC,Record_identifier,Record_identifier_desc,Status_of_system,Status_of_system_desc,Latitude,Longitude,Maximum_sustained_wind_in_knots,Minimum_Pressure_in_millibars,F34_kt_wind_radii_maximum_northeastern,F34_kt_wind_radii_maximum_southeastern,F34_kt_wind_radii_maximum_southwestern,F34_kt_wind_radii_maximum_northwestern,F50_kt_wind_radii_maximum_northeastern,F50_kt_wind_radii_maximum_southeastern,F50_kt_wind_radii_maximum_southwestern,F50_kt_wind_radii_maximum_northwestern,F64_kt_wind_radii_maximum_northeastern,F64_kt_wind_radii_maximum_southeastern,F64_kt_wind_radii_maximum_southwestern,F64_kt_wind_radii_maximum_northwestern
    return res

def process_header(data):
    data=data.split(',')
    Basin,ATCF_cyclone_number_for_that_year,Year,Name,Number_of_best_track_entries=data[0][0:2],data[0][2:4],data[0][4:8],data[1].strip(),data[2].strip()
    res=Basin,ATCF_cyclone_number_for_that_year,Year,Name,Number_of_best_track_entries
    return res


def identify_line_type(data):
    print(data.split(','))
    if len(data.split(','))>4:
        return 2
    else:
        return 1
def columns_name():
    res=['Basin','ATCF_cyclone_number_for_that_year','Year_','Name',
         #'Number_of_best_track_entries',
         'Year','Month','Day','Hours_in_UTC','Minutes_in_UTC',
         'Record_identifier','Record_identifier_desc','Status_of_system','Status_of_system_desc','Latitude','Longitude'
         ,'Maximum_sustained_wind_in_knots','Minimum_Pressure_in_millibars','F34_kt_wind_radii_maximum_northeastern',
         'F34_kt_wind_radii_maximum_southeastern','F34_kt_wind_radii_maximum_southwestern',
         'F34_kt_wind_radii_maximum_northwestern','F50_kt_wind_radii_maximum_northeastern',
         'F50_kt_wind_radii_maximum_southeastern','F50_kt_wind_radii_maximum_southwestern',
         'F50_kt_wind_radii_maximum_northwestern','F64_kt_wind_radii_maximum_northeastern',
         'F64_kt_wind_radii_maximum_southeastern','F64_kt_wind_radii_maximum_southwestern',
         'F64_kt_wind_radii_maximum_northwestern']
    return res

pf=[]
header_fields=[]
filepath = 'hurdat2-1851-2018-051019.txt'
with open(filepath) as fp:
    ln = fp.readline()
    while ln:
        
        lt=identify_line_type(ln)

        details=[]
        if (lt==1):
            header_fields=process_header(ln)
            details=[]
        else:
            details=process_details(ln)
        if (details!=[]):
            n=list(header_fields[:-1])+list(details)

            pf.append(n)
        ln=fp.readline()
        
df=pd.DataFrame(pf)
df.columns=columns_name()
df.to_csv('hur.csv')
print(df.shape)