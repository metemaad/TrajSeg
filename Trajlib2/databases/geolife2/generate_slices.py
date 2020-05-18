import pandas as pd
df=pd.read_csv('geolife_w_features.csv')
print(df.shape,df.columns)
x=[21,154,111,69,73,75,102,154,129,170]
print(list(set(df.t_user_id.values)))
i=1
for _ in x:
    print(_,df.loc[df.t_user_id==_].shape)
    #df2=df.loc[df.t_user_id==_].copy()
    #df2.to_csv('geolife'+str(i)+'.csv')
    i=i+1