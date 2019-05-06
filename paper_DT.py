# Import required packages
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.tree import DecisionTreeRegressor, export_graphviz, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

##################################

#CLASSIFICATOR

##################################

## READING DATA ##

# Setup pandas options
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.5f}'.format

# Get the data
PT_data_1000 = pd.read_excel("../../Data/PTResults-1000.xlsx")

target = PT_data_1000.iloc[:,5:180]
X = PT_data_1000.drop(target, axis=1)  # Just preparing the data
y = target

##################################

## SHUFFLING THE TRAINING/TESTING SPLIT AND GETTING THE BEST TRAINED TREE ##

ncells = 175 # Number of abscissa's points
n0 = 0. # Origin of abscissa for this data


training_features, testing_features, training_target, testing_target = train_test_split(
	X, y, test_size=0.3, shuffle=True)
# store

abscissa = []
ncells_missclass_3 = []
ncells_missclass_6 = []
ncells_missclass_9 = []

##################################

## TRAINING THE DECISION TREE ##

for j in range(ncells):

	abscissa.append(j)

	print(j)

	# Training with Decision Tree
	model_3 = DecisionTreeClassifier(max_depth=3)
	model_3.fit(training_features, training_target)
	model_6 = DecisionTreeClassifier(max_depth=6)
	model_6.fit(training_features, training_target)
	model_9 = DecisionTreeClassifier(max_depth=9)
	model_9.fit(training_features, training_target)

	# Making predictions on the testing features set
	prediction_3 = model_3.predict(testing_features)
	prediction_6 = model_6.predict(testing_features)
	prediction_9 = model_9.predict(testing_features)
	lst3 = [item[j] for item in prediction_3]
	lst6 = [item[j] for item in prediction_6]
	lst9 = [item[j] for item in prediction_9]

	testing_target = np.array(testing_target)
	lst3t = [item[j] for item in testing_target]

	# Number of missclassified cells
	ncells_missclass_3.append(int(300 * mean_squared_error(lst3, lst3t)))
	ncells_missclass_6.append(int(300 * mean_squared_error(lst6, lst3t)))
	ncells_missclass_9.append(int(300 * mean_squared_error(lst9, lst3t)))

abscissa = n0 + np.array(abscissa)*(1.0-n0)/ncells
misclass_3 = np.array(ncells_missclass_3) / 300 * 100 # %
misclass_6 = np.array(ncells_missclass_6) / 300 * 100 # %
misclass_9 = np.array(ncells_missclass_9) / 300 * 100 # %

##################################

## PLOTTING RESULTS ##

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, misclass_3,color='green',label='Tree depth 3')
plt.plot(abscissa, misclass_6,color='red',label='Tree depth 6')
plt.plot(abscissa, misclass_9,label='Tree depth 9')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Misclassifications} [\%]',fontsize=16)
plt.legend()

plt.show()

##################################

## CHECKING TRAINING AND TESTING DISTRIBUTIONS ##

fig1, axes = plt.subplots(1,2)
axes[0].hist(training_features['Temperture[K]'], bins=20)
axes[1].hist(testing_features['Temperture[K]'], bins=20)

fig2, axes1 = plt.subplots(1,2)
axes1[0].hist(training_features['AoA[o]'], bins=20)
axes1[1].hist(testing_features['AoA[o]'], bins=20)

fig3, axes2 = plt.subplots(1,2)
axes2[0].hist(training_features['rho[kg/m3]'], bins=20)
axes2[1].hist(testing_features['rho[kg/m3]'], bins=20)

fig4, axes3 = plt.subplots(1,2)
axes3[0].hist(training_features['MVD[mum]'], bins=20)
axes3[1].hist(testing_features['MVD[mum]'], bins=20)

fig5, axes4 = plt.subplots(1,2)
axes4[0].hist(training_features['Mach'], bins=20)
axes4[1].hist(testing_features['Mach'], bins=20)
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

target = PT_data_1000.iloc[:,5:180]
X = PT_data_1000.drop(target, axis=1)  # Just preparing the data
y = target

##################################

## SHUFFLING THE TRAINING/TESTING SPLIT AND GETTING THE BEST TRAINED TREE ##

ncells = 175 # Number of abscissa's points
n0 = 0. # Origin of abscissa for this data


training_features, testing_features, training_target, testing_target = train_test_split(
	X, y, test_size=0.3, shuffle=True)
# store

abscissa = []
ncells_missclass_3 = []
ncells_missclass_6 = []
ncells_missclass_9 = []

##################################

## TRAINING THE DECISION TREE ##

for j in range(ncells):

	abscissa.append(j)

	print(j)

	# Training with Decision Tree
	model_3 = DecisionTreeRegressor(max_depth=3)
	model_3.fit(training_features, training_target)
	model_6 = DecisionTreeRegressor(max_depth=6)
	model_6.fit(training_features, training_target)
	model_9 = DecisionTreeRegressor(max_depth=9)
	model_9.fit(training_features, training_target)

	# Making predictions on the testing features set
	prediction_3 = model_3.predict(testing_features)
	prediction_6 = model_6.predict(testing_features)
	prediction_9 = model_9.predict(testing_features)
	lst3 = [item[j] for item in prediction_3]
	lst6 = [item[j] for item in prediction_6]
	lst9 = [item[j] for item in prediction_9]

	testing_target = np.array(testing_target)
	lst3t = [item[j] for item in testing_target]

	# Number of missclassified cells
	ncells_missclass_3.append(mean_squared_error(lst3, lst3t))
	ncells_missclass_6.append(mean_squared_error(lst6, lst3t))
	ncells_missclass_9.append(mean_squared_error(lst9, lst3t))

abscissa = n0 + np.array(abscissa)*(1.0-n0)/ncells
misclass_3 = np.array(ncells_missclass_3) * 100 # %
misclass_6 = np.array(ncells_missclass_6) * 100 # %
misclass_9 = np.array(ncells_missclass_9) * 100 # %

##################################

## PLOTTING RESULTS ##

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, misclass_3,color='green',label='Tree depth 3')
plt.plot(abscissa, misclass_6,color='red',label='Tree depth 6')
plt.plot(abscissa, misclass_9,label='Tree depth 9')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{MSE} [\%]',fontsize=16)
plt.legend()

plt.show()

##################################

## CHECKING TRAINING AND TESTING DISTRIBUTIONS ##

fig1, axes = plt.subplots(1,2)
axes[0].hist(training_features['Temperture[K]'], bins=20)
axes[1].hist(testing_features['Temperture[K]'], bins=20)

fig2, axes1 = plt.subplots(1,2)
axes1[0].hist(training_features['AoA[o]'], bins=20)
axes1[1].hist(testing_features['AoA[o]'], bins=20)

fig3, axes2 = plt.subplots(1,2)
axes2[0].hist(training_features['rho[kg/m3]'], bins=20)
axes2[1].hist(testing_features['rho[kg/m3]'], bins=20)

fig4, axes3 = plt.subplots(1,2)
axes3[0].hist(training_features['MVD[mum]'], bins=20)
axes3[1].hist(testing_features['MVD[mum]'], bins=20)

fig5, axes4 = plt.subplots(1,2)
axes4[0].hist(training_features['Mach'], bins=20)
axes4[1].hist(testing_features['Mach'], bins=20)
plt.show()

