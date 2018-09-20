#gaussian.py

#analise estatistica:
#https://machinelearningmastery.com/time-series-data-stationary-python/

import pandas as pd	
from pandas import Series
from pandas import read_csv
from matplotlib import pyplot
series = Series.from_csv('../../Downloads/gams.txt',header=0)
series.hist()
pyplot.show()


	
#from pandas import Series
from statsmodels.tsa.stattools import adfuller
series = pd.read_csv('../../Downloads/gams.txt', header=0,usecols=[1])

X = series.values
print X
X=X.flatten()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
