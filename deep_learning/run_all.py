import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython.utils import io
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_validate
import concurrent.futures
import csv


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in=6, H=10, D_out=2):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 10)
        self.linear3 = torch.nn.Linear(10, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h_relu).sigmoid()
        y_pred = self.linear3(h2_relu)
        return y_pred

## Define the net to work with skorch
net = NeuralNetClassifier(
    TwoLayerNet,
    max_epochs=30,
    lr=0.01,
    criterion = torch.nn.CrossEntropyLoss,
    optimizer = torch.optim.Adam,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    callbacks = []
)


# Data
PT_data_complete = pd.read_excel("../PTResults-1000.xlsx")
# clean up
PT_data_complete = PT_data_complete.drop(columns = 'Unnamed: 0')
print(PT_data_complete.columns)

PT_data = pd.read_excel("../PTResults-1000.xlsx")
ices = np.zeros(175*1000)
for i in range(len(PT_data)):
    ice = PT_data.loc[i].iloc[7:len(PT_data.columns)]
    ices[i*175:(i+1)*175] = ice.values

PT_data = PT_data.iloc[np.repeat(np.arange(len(PT_data)), 175)]
drop_icol = list(range(6,len(PT_data.columns)))

PT_data = PT_data.drop(PT_data.columns[drop_icol],axis=1)
PT_data = PT_data.drop(columns = 'Unnamed: 0')
PT_data['loc'] = np.array(list(range(175))*1000)
PT_data['ice'] = ices
print(PT_data.columns)


# No need for parallel

PT_numpy = PT_data.values
X = PT_numpy[:,:-1]
y = PT_numpy[:,-1]
X = X.astype(np.float32)
y = y.astype(np.int64)  

#testlog = 'results_run_all.csv'
#testcolumns = ['abcsissa','precision','recall']
#with open(testlog,'w') as f:
  #  logger = csv.DictWriter(f, testcolumns)
 #   logger.writeheader()

dict_score = cross_validate(net, X, y, 
    scoring = ['precision', 'recall'], 
    return_estimator = True,
    cv=10)
estimator = dict_score['estimator']
torch.save(dict_score
    , 'model.pth.tar')
#recall = cross_val_score(net, X, y, scoring = 'recall', cv=10)

#with open(testlog,'a') as f:
 #       print('writing')
  #      logger = csv.DictWriter(f, testcolumns)
   #     logger.writerow({'abcsissa':PT_data_complete.columns[i],
    #        'precision':precision,
     #       'recall':recall})

print(dict_score)


