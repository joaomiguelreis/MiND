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
from sklearn.model_selection import cross_val_score
import concurrent.futures
import csv







class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in=5, H=10, D_out=2):
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
    max_epochs=50,
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

def PT_data_generator(label_col):
    '''
    Args:
        label_col (int, >4, <181): column to be considered the label
    '''
    # Make sure the argument is correct
    assert 4<label_col<181
    
    # Create the new dataframe
    PT_data = PT_data_complete.copy()
    
    # Create the column with label
    col_name = PT_data.columns[label_col]
    PT_data['label'] = PT_data[col_name]
    
    # Delete all not needed columns
    PT_data = PT_data.drop(columns = PT_data.columns[5:-1])
    return PT_data

PT_data = PT_data_generator(132)
print(PT_data)


# Parallel?
def procedure(i):  
    print(i)   
    PT_data = PT_data_generator(i)
    PT_numpy = PT_data.values
    X = PT_numpy[:,:-1]
    y = PT_numpy[:,-1]
    X = X.astype(np.float32)
    y = y.astype(np.int64)            
    with io.capture_output() as captured:
        precision = cross_val_score(net, X, y, scoring = 'precision', cv=10)
        recall = cross_val_score(net, X, y, scoring = 'recall', cv=10)
    with open(testlog,'a') as f:
        print('writing')
        logger = csv.DictWriter(f, testcolumns)
        logger.writerow({'abcsissa':PT_data_complete.columns[i],
            'precision_mean':precision.mean(),
            'precision_std':precision.std(),
            'recall_mean':recall.mean(),
            'recall_std':recall.std()
            })
    return precision, recall

testlog = 'results_run.csv'
testcolumns = ['abcsissa','precision_mean','precision_std', 'recall_mean', 'recall_std']
with open(testlog,'w') as f:
	logger = csv.DictWriter(f, testcolumns)
	logger.writeheader()

precision_mean = [] 
precision_std = []
recall_mean = [] 
recall_std = []

with concurrent.futures.ProcessPoolExecutor() as executor:
        for precision, recall in executor.map(procedure, range(5, 181)):
        	precision_mean.append(precision.mean())
        	precision_std.append(precision.std())
        	recall_mean.append(recall.mean())
        	recall_std.append(recall.std())


