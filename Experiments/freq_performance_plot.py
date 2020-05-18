import numpy as np
freq=[1363.8514171557,6343.528957528957,29923.636363636364]
label=['Geolife','Fishing','Hurricanes']
best_performance_C5=np.array([89.823187,80.096490,85.350681])
best_performance_C7=np.array([89.662010,81.534877,87.793770])
best_performance_C9=np.array([90.530032,83.032717,85.626858])
best_performance_C11=np.array([91.062992,84.025796,84.902803])
best_performance_C13=np.array([91.062992,84.025796,84.902803])

best_performance_LR5=np.array([88.859068,90.437149,84.631261])
best_performance_LR7=np.array([88.281470,89.916897,86.252890])
best_performance_LR9=np.array([89.920582,90.423809,85.940859])
best_performance_LR11=np.array([90.164121,91.245083,85.521549])
best_performance_LR13=np.array([90.052490,91.704226,85.856418])


best_performance_RW5=np.array([89.473423,90.643328,86.685047])
best_performance_RW7=np.array([89.797258,86.688494,86.182374])
best_performance_RW9=np.array([90.318693,86.956257,85.167341])
best_performance_RW11=np.array([90.376372,86.961656,83.963985])
best_performance_RW13=np.array([91.090839,85.418624,83.982581])

best_performance_K=np.array([92.325081,83.227268,82.902660 ])
best_performance_L=np.array([88.426831,90.580613,85.433148])

m=(best_performance_K+best_performance_L+best_performance_LR13+
   best_performance_LR11+
best_performance_LR9+
best_performance_LR7+
best_performance_LR5+
   best_performance_C7+best_performance_C9+best_performance_C11+
   best_performance_C13+best_performance_C5+
   best_performance_RW5+
   best_performance_RW7+
   best_performance_RW9+
   best_performance_RW11+
   best_performance_RW13)/17.
print(m)

from matplotlib import pyplot as plt
plt.plot(freq,best_performance_K,c='gray',alpha=0.5)
plt.plot(freq,best_performance_L,c='gray',alpha=0.5)
plt.plot(freq,best_performance_C5,c='gray',alpha=0.5)
plt.plot(freq,best_performance_C7,c='gray',alpha=0.5)
plt.plot(freq,best_performance_C9,c='gray',alpha=0.5)
plt.plot(freq,best_performance_C11,c='gray',alpha=0.5)
plt.plot(freq,best_performance_C13,c='gray',alpha=0.5)
plt.plot(freq,best_performance_LR5,c='gray',alpha=0.5)
plt.plot(freq,best_performance_LR7,c='gray',alpha=0.5)
plt.plot(freq,best_performance_LR9,c='gray',alpha=0.5)
plt.plot(freq,best_performance_LR11,c='gray',alpha=0.5)
plt.plot(freq,best_performance_LR13,c='gray',alpha=0.5)
plt.plot(freq,best_performance_RW5,c='gray',alpha=0.5)
plt.plot(freq,best_performance_RW7,c='gray',alpha=0.5)
plt.plot(freq,best_performance_RW9,c='gray',alpha=0.5)
plt.plot(freq,best_performance_RW11,c='gray',alpha=0.5)
plt.plot(freq,best_performance_RW13,c='gray',alpha=0.5)
ax=plt.plot(freq,m,c='b',linestyle='--',linewidth=4)
plt.annotate('Geolife',xy=(55, 100), xycoords='figure points')
plt.annotate('f='+str(int(freq[0]))+'s',xy=(55, 80), xycoords='figure points')

plt.annotate('Fishing',xy=(115, 100), xycoords='figure points')
plt.annotate('f='+str(int(freq[1]))+'s',xy=(115, 80), xycoords='figure points')

plt.annotate('Hurricanes',xy=(370, 100), xycoords='figure points')
plt.annotate('f='+str(int(freq[2]))+'s',xy=(370, 80), xycoords='figure points')

plt.ylim([60,101])
plt.axvspan(xmin=freq[0]-57,xmax=freq[0]+57,  color='r',  alpha=0.3)
plt.axvspan(xmin=freq[1]-449,xmax=freq[1]+449, color='r',   alpha=0.3)
plt.axvspan(xmin=freq[2]-3347,xmax=freq[2]+3347, color='r',   alpha=0.3)
plt.ylabel('Harmonic mean')
plt.xlabel('The period of data capturing in seconds')
plt.savefig('compare_freq_perf.png')

plt.show()
print(m)
print(np.corrcoef(freq,m)[1][0])
r=np.corrcoef(freq,m)[1][0]



import numpy as np
from minepy import MINE

def print_stats(mine):
    print("Maximal Information Coefficient :", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
    print("GMIC", mine.gmic())
    print("TIC", mine.tic())
from minepy import MINE
x =np.array( freq)
y =np.array( m)
from minepy import MINE

mine = MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(x, y)
from scipy.spatial import distance
import dcor

print("dcor:", dcor.distance_stats(x, y))

print("distance corrolation:", distance.correlation(x, y))
print("Maximal Information Coefficient :", mine.mic())
#print(_, np.corrcoef(__['Harmonic mean'].values, __['ws'].values)[1][0])

