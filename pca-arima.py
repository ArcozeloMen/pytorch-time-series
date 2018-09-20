# PCA-ARIMA 

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import sklearn
from  sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_csv = pd.read_csv('../../Downloads/gams.txt',header=0, usecols=[1,2,3,4,5,6])
#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]

#plt.plot(data_csv)
#plt.legend(('humidity','pm10','temperature','voc','pm25'),loc='upper right')
#plt.show()

#print data_csv
data_csv = StandardScaler().fit_transform(data_csv)
#PM
#data_csv = np.delete(data_csv,[0,1,2,5,6],1)

#Temp-CO2
data_csv=np.delete(data_csv,[0,2,3,4,6],1)
fun_pca=PCA(n_components=1)
data_csv=fun_pca.fit_transform(data_csv)

train=data_csv[:-5000]
teste=data_csv[-5000:-4950]
history=[x for x in train]
#pint(model_fit.summary())
predictions=list()
for i in range(len(teste)):
	model = ARIMA(history, order=(5,1,1))
	model_fit = model.fit(disp=0)
	output=model_fit.forecast()
	yhat=output[0]
	predictions.append(yhat)
	obs=teste[i]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
	plt.plot(i,yhat, 'ro')
	plt.plot(i,obs,'bo')

plt.show()
error = mean_squared_error(teste, predictions)
print('Test MSE: %.3f' % error)

#residuals = DataFrame(model_fit.resid)
#residuals.plot()
#plt.show()
#residuals.plot(kind='kde')
#plt.show()
#print(residuals.describe())

#Divide entre teste e treino

#pm25=train['pm25']

