# v4 - versao supervisionada com shift no target
# - normalizado
# - Estandardizado

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

data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[3,4])
#data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]

data_csv = StandardScaler().fit_transform(data_csv)
#plt.plot(data_csv)
#plt.legend(('humidity','pm10','temperature','voc','pm25'),loc='upper right')
#plt.show()

class Rede(nn.Module):
    def __init__(self):
        super(Rede,self).__init__()
        self.camada1=nn.Linear(1,10)
        self.camada2=nn.Linear(10,1)

    def forward(self, x):
        x=F.relu(self.camada1(x))
        x=F.relu(self.camada2(x))
        return x
#PCA
fun_pca=PCA(n_components=1)
data_csv=fun_pca.fit_transform(data_csv)

#Divide entre teste e treino
train=data_csv[:100000]
teste=data_csv[100000:]



#DataFrame para Tensor
train=torch.tensor(train)
teste=torch.tensor(teste)

#normalizacao
n_train=nn.functional.normalize(train)
n_teste=nn.functional.normalize(teste)

n_train=n_train.float()
#n_train.resize(130099,4)
#print n_train

#	TREINO
print '###	TREINO	###'
_rede=Rede()
optimizer = optim.SGD(_rede.parameters(), lr=0.01)
perda= nn.MSELoss()
for i in range (1):
	for j in range(len(n_train)):
		
		output=torch.zeros([4,1])
		output[0,0]=_rede(n_train[j])
		print output[0][0]
		print output[0,0].item
		print output[0]
		print output
		print _rede(output[0][0])
		output[1,0]=_rede(output[0,0])
		output[2,0]=_rede(output[1,0])
		output[3,0]=_rede(output[2,0])

		output=torch.cat(output,_rede(output))
		#for k in range(4):
			#output=torch.cat(output,_rede(output[-1]))			
			
		target=n_train[j:j+4]
#		print output.shape
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
