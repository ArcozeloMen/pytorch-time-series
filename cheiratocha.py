import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("always")
data_csv = pd.read_csv('../../Downloads/gams.txt', usecols=[2,3,4,5,6])
data_csv= data_csv[['humidity','pm10','temperature','voc','pm25']]
_pm25=data_csv['pm25']
print "####################"
#_a=data_csv.as_matrix()
_b=data_csv[:,:4]

_a=_a[:,4:5]
print _a
print _b
plt.plot(data_csv)

plt.legend(('humidity','pm10','temperature','voc','pm25'),loc='upper right')
plt.show()

class Rede(nn.Module):
    def __init__(self):
        super(Rede,self).__init__()
        self.camada1=nn.Linear(4,10)
        self.camada2=nn.Linear(10,1)

    def forward(self, x):
        x=self.camada1(x)
        x=self.camada2(x)
        return x

#_rede=Rede()
#output= _rede(data_csv)

train=data_csv[:-5000]
teste=data_csv[-5000:]

_a=torch.from_numpy(_b)
train=torch.tensor(_b.values)
col_erro=torch.tensor(_a)
teste=torch.tensor(teste.values)
ntrain=nn.functional.normalize(train)

nteste=nn.functional.normalize(teste)
ntrain.resize(130099,4)
print ntrain

_rede=\
Rede()
output=_rede(ntrain.float())
#print output
   
