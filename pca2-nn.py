# PCA2-NN.py 


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("always")
data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[1,2,3,4,5,6])

#plt.plot(data_csv)
#plt.legend(('co2','humidity','pm10','temperature','voc','pm25'),loc='upper right')
#plt.show()

class Rede(nn.Module):
    def __init__(self):
        super(Rede,self).__init__()
        self.camada1=nn.Linear(1,10)
        self.camada2=nn.Linear(10,1)

		

    def forward(self, x):
        x=self.camada1(x)
        x=self.camada2(x)
        return x

#Divide entre teste e treino
train=data_csv[:-5000]
teste=data_csv[-5000:]


#Estandardizaçao
train = StandardScaler().fit_transform(train)

#PM25 para testar em vez do PCA como output
pm25=np.delete(train,[0,1,2,4,5,6],1)
pm25=np.roll(pm25,-1)
pm25=torch.tensor(pm25)
pm25=pm25.resize(130099,1)
pm25=nn.functional.normalize(pm25)

#PCA
reducao=np.delete(train,[0,1,4,5],1)


reducaoPCA=PCA(n_components=1)
reducao=reducaoPCA.fit_transform(reducao)
reducao=torch.from_numpy(reducao)
print (reducao.size())


#pm25=train['pm25']

#Tirar o pm25 do treino/teste
#train=train.drop(columns=['pm25'])
#teste=teste.drop(columns=['pm25'])

#DataFrame para Tensor
#train=torch.tensor(train.values)
#teste=torch.tensor(teste.values)
#pm25=torch.tensor(pm25.values)
#pm25=pm25.resize(130099,1)
#pm25=pm25.float()
#train.unsqueeze_(-1)

#train=train.transpose(2,1)
#train=train.expand(130099,1,4)
#train=train.resize(130099,20,4)


#normalizacao
reducao=nn.functional.normalize(reducao)
#n_teste=nn.functional.normalize(teste)
#pm25=nn.functional.normalize(pm25)

#n_train.resize(130099,4)
#print n_train
#print n_train.shape

#	TREINO
print ('###	  TREINO  	###')
_rede=Rede()
optimizer = optim.SGD(_rede.parameters(), lr=0.01)
perda= nn.MSELoss()
target=np.roll(reducao,-1)
target=torch.from_numpy(target)
target=nn.functional.normalize(target)

olha=torch.zeros([130099,1],dtype=torch.float)
olha.random_(-2,2)

for i in range (50):
	output=_rede(olha)
	#output=np.asarray(output)
	#output=torch.from_numpy(output)
	#output=torch.Tensor(output(dtype='float'))
	#output.ToTensor()
	#output=list(output)
	#output=np.asarray(output)
	
	#output=torch.squeeze(output,2)
	#output=output.reshape(130099,1)
	#target=pm25	
	#print output.shape
	erro=perda(output,pm25.float())
	optimizer.zero_grad()
	erro.backward()
	optimizer.step()
	error = str(erro)
	error=error.split('(')
	error=error[1].split(',')
	print (error[0])
	plt.plot(i,float(error[0]),'bo')
	#print output
plt.show()
