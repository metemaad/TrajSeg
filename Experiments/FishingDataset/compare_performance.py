import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Performance_FishingDataset_SWS_LR_WS_11.csv")
df2=pd.read_csv("Performance_FishingDataset_SPD.csv")
df3=pd.read_csv("Performance_FishingDataset_SWS_RW_WS_7.csv")
df4=pd.read_csv("Performance_FishingDataset_SWS_C_WS_7.csv")
df5=pd.read_csv("Performance_FishingDataset_CBSMoT.csv")
df6=pd.read_csv("Performance_FishingDataset_SWS_L.csv")
df7=pd.read_csv("Performance_FishingDataset_SWS_C_WS_11.csv")
#df8=pd.read_csv("Performance_FishingDataset_CBSMoT.csv")
df9=pd.read_csv("Performance_FishingDataset_SWS_K.csv")
#df10=df9
df11=pd.read_csv("Performance_FishingDataset_WKMeans.csv")
df12=pd.read_csv("Performance_FishingDataset_GRASPUTS.csv")





df['memory']=df['memory']-df['memory'][0]
df2['memory']=df2['memory']-df2['memory'][0]
df3['memory']=df3['memory']-df3['memory'][0]
df4['memory']=df4['memory']-df4['memory'][0]
df5['memory']=df5['memory']-df5['memory'][0]
df6['memory']=df6['memory']-df6['memory'][0]
df7['memory']=df7['memory']-df7['memory'][0]
#df8['memory']=df8['memory']-df8['memory'][0]
df9['memory']=df9['memory']-df9['memory'][0]
df12['memory']=df12['memory']-df12['memory'][0]
df11['memory']=df11['memory']-df11['memory'][0]

plt.plot(df['time'],df['memory'],'b')
plt.plot(df2['time'],df2['memory'],'r')
plt.plot(df3['time'],df3['memory'],'g')
plt.plot(df4['time'],df4['memory'],'y')
plt.plot(df5['time'],df5['memory'],'k')
plt.plot(df6['time'],df6['memory'],'gray')
plt.plot(df7['time'],df7['memory'],'pink')
#plt.plot(df8['time'],df8['memory'],'orange')
plt.plot(df9['time'],df9['memory'],'darkblue')
#plt.plot(df10['time'],df10['memory'],'darkred')
plt.plot(df11['time'],df11['memory'],'gray')
plt.plot(df12['time'],df12['memory'],'darkred')
plt.ylabel('Memory in MB')
plt.xlabel('CPU Process time in seconds')




plt.legend(['SWS_LR WS_11','SPD','SWS_RW WS_7','SWS_C WS_7',
            'CBSMoT','SWS_L','SWS_C WS_11','SWS_K','WKMeans','GRASPUTS'])
#plt.legend([])
plt.scatter(df['time'][-1:],df['memory'][-1:],s=200,c='b')
#plt.axvline(x=df['time'][-1:].values[0],ymin=0,ymax=0.5,color='b',linestyle=':')
#plt.axhline(y=df['memory'][-1:].values[0],xmin=0,xmax=0.5,color='b',linestyle=':')
plt.scatter(df2['time'][-1:],df2['memory'][-1:],s=200,c='r')
plt.scatter(df3['time'][-1:],df3['memory'][-1:],s=200,c='g')
plt.scatter(df4['time'][-1:],df4['memory'][-1:],s=200,c='y')
plt.scatter(df5['time'][-1:],df5['memory'][-1:],s=200,c='k')
#plt.axvline(x=df5['time'][-1:].values[0],ymin=0,ymax=0.95,color='k',linestyle=':')
#plt.axhline(y=df5['memory'][-1:].values[0],xmin=0,xmax=0.95,color='k',linestyle=':')
plt.scatter(df6['time'][-1:],df6['memory'][-1:],s=200,c='gray')
plt.scatter(df7['time'][-1:],df7['memory'][-1:],s=200,c='pink')
#plt.scatter(df8['time'][-1:],df8['memory'][-1:],s=200,c='orange')
plt.scatter(df9['time'][-1:],df9['memory'][-1:],s=200,c='darkblue')
#plt.scatter(df10['time'][-1:],df10['memory'][-1:],s=200,c='darkred')
plt.scatter(df11['time'][-1:],df11['memory'][-1:],s=200,c='gray')
plt.scatter(df12['time'][-1:],df12['memory'][-1:],s=200,c='darkred')
plt.xlim([0,170])
plt.ylim([0,6])
plt.title("Fishing dataset")
#plt.ylim([0,15])
plt.savefig("Fishing_cpm_performance.png")
plt.show()