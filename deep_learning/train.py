# Import general python packages
import csv
import concurrent.futures
from IPython.utils import io

# Import ML/DL related packages
from pandas import read_excel
from torch import save
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score, cross_validate

# Import auxiliary functions
from models import TwoLayerNet, TwoLayerNetMult
from data_generators import Label_Gen, mult_label_gen
from argparser import parser

## ARGUMENTS
args = parser.parse_args()

## MODEL
if args.all:
    Net = TwoLayerNetMult
else:
    Net = TwoLayerNet

net = NeuralNetClassifier(
    Net,
    max_epochs=args.epochs,
    lr=args.lr,
    criterion = CrossEntropyLoss,
    optimizer = Adam,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    callbacks = []
)

## DATA
PT_data_complete = read_excel(args.datadir)
# clean up
PT_data_complete = PT_data_complete.drop(columns = 'Unnamed: 0')
if args.all:
    X,y = mult_label_gen(PT_data_complete)
else:
    PT_data = Label_Gen(PT_data_complete)

## TRAIN MODEL
if args.all:
    # One model for all ice regions
    dict_score = cross_validate(net, X, y, 
            scoring = ['precision', 'recall'], 
            return_estimator = True, cv=10)
    estimator = dict_score['estimator']
    save({'estimator': estimator}, 
            args.logdir+'model.pth.tar')
else:
    # One model for each ice region
    def procedure(i):  
        print('Processing model number ',i)   
        X,y = PT_data[i]        
        with io.capture_output() as captured:
            precision = cross_val_score(net, X, y, scoring = 'precision', cv=10)
            recall = cross_val_score(net, X, y, scoring = 'recall', cv=10)
        with open(testlog,'a') as f:
            logger = csv.DictWriter(f, testcolumns)
            logger.writerow({'abcsissa':PT_data_complete.columns[i],
                'precision_mean':precision.mean(),
                'precision_std':precision.std(),
                'recall_mean':recall.mean(),
                'recall_std':recall.std()
                })
        return precision, recall

    testlog = args.logdir + 'scores.csv'
    testcolumns = ['abcsissa','precision_mean','precision_std', 'recall_mean', 'recall_std']
    with open(testlog,'w') as f:
    	logger = csv.DictWriter(f, testcolumns)
    	logger.writeheader()

    with concurrent.futures.ProcessPoolExecutor() as executor:
            for precision, recall in executor.map(procedure, range(5, 181)):
            	pass


