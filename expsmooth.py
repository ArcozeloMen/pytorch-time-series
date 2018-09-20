import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[6])
#pm=pd.read_csv('../../Downloads/gams.txt', usecols=[3,6])
#data_csv = StandardScaler().fit_transform(data_csv)

#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]
#pm =np.delete(data_csv,[2,4,6],1)
#pm =np.delete(pm,3,1) 



#pm2 = PCA(n_components=1)
#pm2= pm2.fit_transform(data_csv)
#for i in range (len(pm)):
#	print(pm[i])  
plt.plot(data_csv)

plt.legend(('voc'),loc='upper right')
plt.show()
data_csv=data_csv.values
data_csv=data_csv.flatten()
#print (data_csv.shape())
y=savgol_filter(data_csv,101,1)
plt.plot(y)
plt.show()

