# Import required packages
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.tree import DecisionTreeRegressor, export_graphviz, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

##################################

#CLASSIFIER

##################################

## READING DATA ##

# Setup pandas options
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.5f}'.format

# Get the data
PT_data_1000 = pd.read_excel("../../Data/PTResults-1000.xlsx")
abscissa = np.loadtxt("../../Data/abscissa.txt")

##################################

## SETTING UP VARIABLES ##

ncells = 176 # Number of abscissa's points

#PRECISION

prec_mean_3 = []
prec_mean_6 = []
prec_mean_9 = []

prec_std_3 = []
prec_std_6 = []
prec_std_9 = []

#RECALL

rec_mean_3 = []
rec_mean_6 = []
rec_mean_9 = []

rec_std_3 = []
rec_std_6 = []
rec_std_9 = []

error_integral_prec = 0.
error_integral_rec = 0.
error_integral_prec_3 = 0.
error_integral_rec_3 = 0.

##################################

## TRAINING THE DECISION TREES ##

for i in range(ncells):

	print(i)

	target = PT_data_1000.iloc[:,5+i]  # PT_data_1000.iloc[:,5:181]
	X = PT_data_1000.iloc[:,0:5]  # PT_data_1000.drop(target, axis=1)  # Remove all columns that are target
	y = target

	# Making predictions on the testing features set with precision metric

	precision_3 = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=10), X, y,
								  scoring = 'precision', cv=10)
	precision_6 = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), n_estimators=10), X, y,
								  scoring = 'precision', cv=10)
	precision_9 = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=10), X, y,
								  scoring = 'precision', cv=10)

	# Making predictions on the testing features set with recall metric
	recall_3 = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=10), X, y,
								  scoring = 'recall', cv=10)
	recall_6 = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=6), n_estimators=10), X, y,
								  scoring = 'recall', cv=10)
	recall_9 = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=10), X, y,
								  scoring = 'recall', cv=10)

	print("Precision: %0.2f (+/- %0.2f)" % (recall_9.mean(), recall_9.std() * 2))

	# Mean and std
	prec_mean_3.append(precision_3.mean())
	prec_mean_6.append(precision_6.mean())
	prec_mean_9.append(precision_9.mean())

	prec_std_3.append(precision_3.std())
	prec_std_6.append(precision_6.std())
	prec_std_9.append(precision_9.std())

	rec_mean_3.append(recall_3.mean())
	rec_mean_6.append(recall_6.mean())
	rec_mean_9.append(recall_9.mean())

	rec_std_3.append(recall_3.std())
	rec_std_6.append(recall_6.std())
	rec_std_9.append(recall_9.std())

	error_integral_prec_3 += precision_3.mean()
	error_integral_rec_3 += recall_3.mean()

	error_integral_prec += precision_9.mean()
	error_integral_rec += recall_9.mean()

prec_error_3 = np.array(prec_mean_3) * 100 # %
prec_error_6 = np.array(prec_mean_6) * 100 # %
prec_error_9 = np.array(prec_mean_9) * 100 # %

rec_error_3 = np.array(rec_mean_3) * 100 # %
rec_error_6 = np.array(rec_mean_6) * 100 # %
rec_error_9 = np.array(rec_mean_9) * 100 # %

prec_errstd_3 = (2*np.array(prec_std_3)) * 100 # %
prec_errstd_6 = (2*np.array(prec_std_6)) * 100 # %
prec_errstd_9 = (2*np.array(prec_std_9)) * 100 # %

rec_errstd_3 = (2*np.array(rec_std_3)) * 100 # %
rec_errstd_6 = (2*np.array(rec_std_6)) * 100 # %
rec_errstd_9 = (2*np.array(rec_std_9)) * 100 # %

print('precision 3 =', error_integral_prec_3, 'recall 3', error_integral_rec_3)
print('precision 9 =', error_integral_prec, 'recall 9', error_integral_rec)

##################################

## PLOTTING RESULTS ##

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, prec_error_3,color='green',label='Tree depth 3')
plt.plot(abscissa, prec_error_6,color='red',label='Tree depth 6')
plt.plot(abscissa, prec_error_9,label='Tree depth 9')
#plt.fill_between(abscissa, prec_error_9-prec_errstd_9, prec_error_9+prec_errstd_9,  # fills area according to standard deviation
		#alpha=0.5,color='#e0e0e0')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Precision} [\%]',fontsize=16)
plt.ylim(-0.5,105)
plt.legend()

plt.show()

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, rec_error_3,color='green',label='Tree depth 3')
plt.plot(abscissa, rec_error_6,color='red',label='Tree depth 6')
plt.plot(abscissa, rec_error_9,label='Tree depth 9')
#plt.fill_between(abscissa, rec_error_9-rec_errstd_9, rec_error_9+rec_errstd_9,  # fills area according to standard deviation
		#alpha=0.5,color='#e0e0e0')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Recall} [\%]',fontsize=16)
plt.ylim(-0.5,105)
plt.legend()

plt.show()

##################################

#REGRESSOR

##################################

## READING DATA ##

# Setup pandas options
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.5f}'.format

# Get the data
PT_data_1000 = pd.read_excel("../../Data/Data_Colleff_Entire.xlsx")

##################################

## SETTING UP VARIABLES ##

ncells = 176 # Number of abscissa's points

#PRECISION

prec_mean_3 = []
prec_mean_6 = []
prec_mean_9 = []

prec_std_3 = []
prec_std_6 = []
prec_std_9 = []

##################################

## TRAINING THE DECISION TREES ##

for i in range(ncells):

	print(i)

	target = PT_data_1000.iloc[:,5+i]  # PT_data_1000.iloc[:,5:181]
	X = PT_data_1000.iloc[:,0:5]  # PT_data_1000.drop(target, axis=1)  # Remove all columns that are target
	y = target

	# Making predictions on the testing features set with precision metric

	precision_3 = cross_val_score(AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=10), X, y,
								  scoring = 'neg_mean_squared_error', cv=10)
	precision_6 = cross_val_score(AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=10), X, y,
								  scoring = 'neg_mean_squared_error', cv=10)
	precision_9 = cross_val_score(AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), n_estimators=10), X, y,
								  scoring = 'neg_mean_squared_error', cv=10)

	print("Precision: %0.2f (+/- %0.2f)" % (recall_9.mean(), recall_9.std() * 2))

	# Mean and std
	prec_mean_3.append(-precision_3.mean())
	prec_mean_6.append(-precision_6.mean())
	prec_mean_9.append(-precision_9.mean())

	prec_std_3.append(precision_3.std())
	prec_std_6.append(precision_6.std())
	prec_std_9.append(precision_9.std())

prec_error_3 = np.array(prec_mean_3)
prec_error_6 = np.array(prec_mean_6)
prec_error_9 = np.array(prec_mean_9)

prec_errstd_3 = (2*np.array(prec_std_3))
prec_errstd_6 = (2*np.array(prec_std_6))
prec_errstd_9 = (2*np.array(prec_std_9))

##################################

## PLOTTING RESULTS ##

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, prec_error_3,color='green',label='Tree depth 3')
plt.plot(abscissa, prec_error_6,color='red',label='Tree depth 6')
plt.plot(abscissa, prec_error_9,label='Tree depth 9')
#plt.fill_between(abscissa, prec_error_9-prec_errstd_9, prec_error_9+prec_errstd_9,
		#alpha=0.5,color='#e0e0e0')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{MSE}',fontsize=16)
plt.legend()

plt.show()