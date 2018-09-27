#v3 - NN
# normalizado
# sem supervizao
# nao faz sentido eliminar um PM dos inputs, dar um PM nos inputs, e prever um PM

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy import concatenate
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("always")
data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[2,3,4,5,6])
data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]

#plt.plot(data_csv)
#plt.legend(('humidity','pm10','temperature','voc','pm25'),loc='upper right')
#plt.show()

class Rede(nn.Module):
    def __init__(self):
        super(Rede,self).__init__()
        self.camada1=nn.Linear(4,2)
        self.camada2=nn.Linear(2,1)

    def forward(self, x):
        x=torch.tanh(self.camada1(x))
        x=self.camada2(x)
        return x

#Divide entre teste e treino
train=data_csv[:-5000]
teste=data_csv[-5000:]
pm25=train['pm10']
pm25=np.roll(pm25,-20)

#Tirar o pm25 do treino/teste
train=train.drop(columns=['pm10'])
teste=teste.drop(columns=['pm10'])


#DataFrame para Tensor

train=torch.tensor(train.values)
teste=torch.tensor(teste.values)
pm25=torch.tensor(pm25)
pm25=pm25.resize(130099,1)

#normalizacao
n_train=nn.functional.normalize(train)
n_teste=nn.functional.normalize(teste)
pm25=nn.functional.normalize(pm25)

#n_train.resize(130099,4)
#print n_train

olha=torch.zeros([130099,4],dtype=torch.float)
olha.random_(-2,2)
misto=torch.zeros([130099,1],dtype=torch.float)
misto.random_()
print olha
#	TREINO
print '###	TREINO	###'
_rede=Rede()
optimizer = optim.SGD(_rede.parameters(), lr=0.01)
perda= nn.MSELoss()
for i in range (50):
	output=_rede(olha)
        target=Variable(pm25)   
 #      print output.shape
        erro=perda(output,target.float())
        optimizer.zero_grad()
        erro.backward()
        optimizer.step()
        error = str(erro)
        error=error.split('(')
        error=error[1].split(',')
        print error[0]
        plt.plot(i,float(error[0]),'bo')
	#print output

plt.show()
