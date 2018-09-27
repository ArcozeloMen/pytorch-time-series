# v7 - ARIMA 
#https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR, DynamicVAR
import matplotlib.pyplot as plt
import sklearn
from  sklearn.metrics import mean_squared_error

data_csv = pd.read_csv('../../Downloads/gams.txt',header=0, usecols=[1,2,3,4,5,6])
#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]

#plt.plot(data_csv)
#plt.legend(('humidity','pm10','temperature','voc','pm25'),loc='upper right')
#plt.show()

train=data_csv[:-5000]
teste=data_csv[-5000:-4950]
history=[x for x in train.values]
#pint(model_fit.summary())


#model=VAR(history)
#results=model.fit(2)
#print results.summary()
#results.plot()
#plt.show()
#results.plot_acorr()
#plt.show()
predictions=list()
for i in range(len(teste)):
	
	#print history[-1]
	model = VAR(history)
	model_fit = model.fit(12)
	lag_oder=model_fit.k_ar
	output=model_fit.forecast(history[-lag_oder:],1)
	yhat=output[0]
	predictions.append(yhat)
	obs=teste.values[i]
	
	#print obs
	history.append(obs)
	#print('predicted=%f, expected=%f' % (yhat[3], obs[3]))
	plt.plot(i,yhat[5],'ro')
	plt.plot(i,obs[5],'bo')

#print model_fit.summary()
teste=np.asarray(teste.values)
predictions=np.asarray(predictions)

print teste.shape
print predictions.shape

teste=np.delete(teste,[0,1,3,4,5],1)
predictions=np.delete(predictions,[0,1,3,4,5],1)

print predictions
print teste
error = mean_squared_error(teste, predictions)
print('Test MSE: %.3f' % error)
plt.show()
#residuals = DataFrame(model_fit.resid)
#residuals.plot()
#plt.show()
#residuals.plot(kind='kde')
#plt.show()
#print(residuals.describe())

#Divide entre teste e treino

#pm25=train['pm25']

