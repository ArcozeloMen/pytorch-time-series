import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler

data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[1,2,3,4,5,6])
#pm=pd.read_csv('../../Downloads/gams.txt', usecols=[3,6])
data_csv = StandardScaler().fit_transform(data_csv)

#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]
pm =np.delete(data_csv,[2,4,6],1)
#pm =np.delete(pm,3,1) 



pm2 = PCA(n_components=1)
pm2= pm2.fit_transform(data_csv)
#for i in range (len(pm)):
#	print(pm[i])  
plt.plot(data_csv)

plt.legend(('co2','humidity','pm10','pm25','temperature','voc',),loc='upper right')
plt.show()


