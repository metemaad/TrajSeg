import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Performance_HurricanesDataset_SWS_C_WS_7.csv")
df2=pd.read_csv("Performance_HurricanesDataset_SPD.csv")
df3=pd.read_csv("Performance_HurricanesDataset_SWS_RW_WS_7.csv")
df4=pd.read_csv("Performance_HurricanesDataset_SWS_C_WS_5.csv")
df5=pd.read_csv("Performance_HurricanesDataset_CBSMoT.csv")
df6=pd.read_csv("Performance_HurricanesDataset_SWS_L.csv")
df7=pd.read_csv("Performance_HurricanesDataset_SWS_C_WS_11.csv")
df8=pd.read_csv("Performance_HurricanesDataset_CBSMoT.csv")
df9=pd.read_csv("Performance_HurricanesDataset_SWS_K.csv")


df['memory']=df['memory']-df['memory'][0]
df2['memory']=df2['memory']-df2['memory'][0]
df3['memory']=df3['memory']-df3['memory'][0]
df4['memory']=df4['memory']-df4['memory'][0]
df5['memory']=df5['memory']-df5['memory'][0]
df6['memory']=df6['memory']-df6['memory'][0]
df7['memory']=df7['memory']-df7['memory'][0]
df8['memory']=df8['memory']-df8['memory'][0]
df9['memory']=df9['memory']-df9['memory'][0]

plt.plot(df['time'],df['memory'],'b')
plt.plot(df2['time'],df2['memory'],'r')
plt.plot(df3['time'],df3['memory'],'g')
plt.plot(df4['time'],df4['memory'],'y')
plt.plot(df5['time'],df5['memory'],'k')
plt.plot(df6['time'],df6['memory'],'gray')
plt.plot(df7['time'],df7['memory'],'pink')
plt.plot(df8['time'],df8['memory'],'orange')
plt.plot(df9['time'],df9['memory'],'darkblue')
plt.ylabel('Memory in MB')
plt.xlabel('CPU Process time in seconds')



plt.legend(['SWS_C WS_7','SPD','SWS_RW WS_7','SWS_C WS_5',
            'CB-SMoT_HT','SWS_L','SWS_C WS_11','CBSMoT','SWS_K'])
plt.scatter(df['time'][-1:],df['memory'][-1:],s=200,c='b')
plt.scatter(df2['time'][-1:],df2['memory'][-1:],s=200,c='r')
plt.scatter(df3['time'][-1:],df3['memory'][-1:],s=200,c='g')
plt.scatter(df4['time'][-1:],df4['memory'][-1:],s=200,c='y')
plt.scatter(df5['time'][-1:],df5['memory'][-1:],s=200,c='k')
plt.scatter(df6['time'][-1:],df6['memory'][-1:],s=200,c='gray')
plt.scatter(df7['time'][-1:],df7['memory'][-1:],s=200,c='pink')
plt.scatter(df8['time'][-1:],df8['memory'][-1:],s=200,c='orange')
plt.scatter(df9['time'][-1:],df9['memory'][-1:],s=200,c='darkblue')
plt.show()