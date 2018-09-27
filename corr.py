import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, normalize
from scipy.stats import pearsonr
warnings.filterwarnings("always")
data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[1,2,3,4,5,6])
#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]


def autocorr(x):
    result =np.correlate(x, x, mode='full')
    return result[result.size/2:]

a=np.array(data_csv[:1000])
#a=StandardScaler().fit_transform(a)
a=normalize(data_csv)
print 'correlation coeficient'
#print autocorr(a.flatten())
print np.corrcoef(a.T)

#print pearsonr(data_csv[0],data_csv[4])
#print pearsonr(data_csv[2],data_csv[3])
#print pearsonr(data_csv[0],data_csv[5])
#print pearsonr(data_csv[0],data_csv[3])
#data_csv=DataFrame(data_csv)


#plt.plot(data_csv)

#plt.legend(('co2','humidity','pm10','pm25','temp','voc'),loc='upper right')
#plt.show()


print 'Correlacao\n'
print data_csv.corr(method='spearman')
print 'Covariancia\n'
print data_csv.cov()
print 'Describe\n'
print data_csv.describe(include='all')
print 'Desvio padrao\n'
print data_csv.std()

data_csv = Series.from_csv('../../Downloads/gams.txt', header=0)
#from sklearn.preprocessing import MinMaxScaler
#data_csv=np.array(data_csv)
#scaler=MinMaxScaler()
#data_csv = StandardScaler().fit_transform(data_csv)
#data_csv=scaler.fit(data_csv)
#data_csv=np.delete(data_csv,[0,1,2,3,4,5],1)
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data_csv[:3000], model='multiplicative')
result.plot()
plt.show()
