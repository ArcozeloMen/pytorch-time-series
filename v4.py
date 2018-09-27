# v4 - versao supervisionada 
# - com shift no target
# - normalizado
# - com conjunto de validacao

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("always")
data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[1,2,3,4,5,6])
#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]

#plt.plot(data_csv)
#plt.legend(('humidity','pm10','temperature','voc','pm25'),loc='upper right')
#plt.show()

class Rede(nn.Module):
    def __init__(self):
        super(Rede,self).__init__()
        self.camada1=nn.Linear(5,10)
        self.camada2=nn.Linear(10,1)

    def forward(self, x):
        x=torch.tanh(self.camada1(x))
        x=torch.tanh(self.camada2(x))
        return x

#Divide entre teste e treino
train=data_csv[:-5000]
teste=data_csv[-5000:-2000]
pm25=train['voc']
pm25=np.roll(pm25,-1)

#print pm25.shape

#Tirar o pm25 do treino/teste
train=train.drop(columns=['voc'])
teste=teste.drop(columns=['voc'])

#DataFrame para Tensor
train=torch.tensor(train.values)
teste=torch.tensor(teste.values)
pm25=torch.tensor(pm25)
pm25=pm25.resize(130099,1)
pm25=pm25.float()

#normalizacao
n_train=nn.functional.normalize(train)
n_teste=nn.functional.normalize(teste)
pm25=nn.functional.normalize(pm25)

#n_train.resize(130099,4)
#print n_train

#	TREINO
print '###	TREINO	###'
_rede=Rede()
optimizer = optim.SGD(_rede.parameters(), lr=0.1)
perda= nn.MSELoss()
for i in range (50):
	output=_rede(n_train.float())
	target=Variable(pm25)	
#	print output.shape
	erro=perda(output,target)
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

pm25=data_csv['voc']
pm25=pm25[-5000:-2000]
pm25=np.roll(pm25,-1)
pm25=torch.tensor(pm25)
pm25=pm25.resize(3000,1)
pm25=nn.functional.normalize(pm25)

olha=torch.zeros([3000,5],dtype=torch.float)
olha.random_(-2,2)

output=_rede(olha)
target=Variable(pm25)
erro=perda(output,target.float())
error = str(erro)
error=error.split('(')
error=error[1].split(',')
print error[0]
plt.plot(i,float(error[0]),'bo')
plt.show()
