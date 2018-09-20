import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("always")
data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[1,2,3,4,5,6])
#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]

plt.plot(data_csv)

plt.legend(('co2','humidity','pm10','pm25','temp','voc'),loc='upper right')
plt.show()

print data_csv.corr()

class Rede(nn.Module):
    def __init__(self):
        super(Rede,self).__init__()
        self.camada1=nn.Linear(4,10)
        self.camada2=nn.Linear(10,1)

    def forward(self, x):
        x=F.relu(self.camada1(x))
        x=F.relu(self.camada2(x))
        return F.relu(x)

#Divide entre teste e treino
train=data_csv[:-5000]
teste=data_csv[-5000:]
pm25=train['pm25']

#Tirar o pm25 do treino/teste
train=train.drop(columns=['pm25'])
teste=teste.drop(columns=['pm25'])

#DataFrame para Tensor
train=torch.tensor(train.values)
teste=torch.tensor(teste.values)
pm25=torch.tensor(pm25.values)
pm25=pm25.resize(130099,1)
pm25=pm25.float()

#normalizacao
n_train=nn.functional.normalize(train)
n_teste=nn.functional.normalize(teste)

#n_train.resize(130099,4)
#print n_train

#	TREINO
print '###	TREINO	###'
for i in range (10):
	_rede=\
	Rede()
	output=_rede(n_train.float())
	target=Variable(pm25)	
	perda= nn.MSELoss()
	erro=perda(output,target)
	_rede.zero_grad()
	erro.backward()
	print erro
	#print output
   
