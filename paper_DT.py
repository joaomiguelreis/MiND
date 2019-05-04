# Import required packages
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.tree import DecisionTreeRegressor, export_graphviz, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

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

training_features_vector = []
testing_features_vector = []
training_target_vector = []
testing_target_vector = []

N = 1000 # Number of shuffles considered
shuffles = 1000
ncells = 175 # Number of abscissa's points
n0 = 0. # Origin of abscissa for this data

for i in range(shuffles):
	training_features, testing_features, training_target, testing_target = train_test_split(
		X, y, test_size=0.15, shuffle=True)
	# store
	training_features_vector.append(training_features)
	testing_features_vector.append(testing_features)
	training_target_vector.append(training_target)
	testing_target_vector.append(testing_target)

minimum_3 = []
minimum_10 = []
minimum_15 = []
abscissa = []
location = []

##################################

## TRAINING THE DECISION TREE ##

for j in range(ncells):

	ncells_missclass_3 = []
	ncells_missclass_10 = []
	ncells_missclass_15 = []

	abscissa.append(j)

	print(j)

	for i in range(N):
		training_features = training_features_vector[i]
		testing_features = testing_features_vector[i]
		training_target = training_target_vector[i]
		testing_target = testing_target_vector[i]

		# Training with Decision Tree
		model_3 = DecisionTreeClassifier(max_depth=3)
		model_3.fit(training_features, training_target)
		model_10 = DecisionTreeClassifier(max_depth=10)
		model_10.fit(training_features, training_target)
		model_15 = DecisionTreeClassifier(max_depth=15)
		model_15.fit(training_features, training_target)

		# Making predictions on the testing features set
		prediction_3 = model_3.predict(testing_features)
		prediction_10 = model_10.predict(testing_features)
		prediction_15 = model_15.predict(testing_features)
		lst3 = [item[j] for item in prediction_3]
		lst10 = [item[j] for item in prediction_10]
		lst15 = [item[j] for item in prediction_15]

		testing_target = np.array(testing_target)
		lst3t = [item[j] for item in testing_target]

		# Number of missclassified cells
		ncells_missclass_3.append(int(150 * mean_squared_error(lst3, lst3t)))
		ncells_missclass_10.append(int(150 * mean_squared_error(lst10, lst3t)))
		ncells_missclass_15.append(int(150 * mean_squared_error(lst15, lst3t)))

	location.append(np.argmin(ncells_missclass_10[0:N]))
	minimum_3.append(min(ncells_missclass_3))
	minimum_10.append(min(ncells_missclass_10))
	minimum_15.append(min(ncells_missclass_15))

print(location)

abscissa = n0 + np.array(abscissa)*(1.0-n0)/ncells
minimum_3 = np.array(minimum_3) / 150 * 100 # %
minimum_10 = np.array(minimum_10) / 150 * 100 # %
minimum_15 = np.array(minimum_15) / 150 * 100 # %

##################################

## PLOTTING RESULTS ##

plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

plt.plot(abscissa, minimum_3,color='green',label='Tree depth 3')
plt.plot(abscissa, minimum_10,color='red',label='Tree depth 10')
plt.plot(abscissa, minimum_15,label='Tree depth 15')

plt.xlabel(r'\textbf{Abscissa}',fontsize=12)
plt.ylabel(r'\textbf{Misclassifications} [\%]',fontsize=16)
plt.legend()

plt.show()
exit(0)

##################################

## CHECKING TRAINING AND TESTING DISTRIBUTIONS ##

fig1, axes = plt.subplots(1,2)
axes[0].hist(training_features_vector[0]['Temperture[K]'], bins=20)
axes[1].hist(testing_features_vector[0]['Temperture[K]'], bins=20)

fig2, axes1 = plt.subplots(1,2)
axes1[0].hist(training_features_vector[0]['AoA[o]'], bins=20)
axes1[1].hist(testing_features_vector[0]['AoA[o]'], bins=20)

fig3, axes2 = plt.subplots(1,2)
axes2[0].hist(training_features_vector[0]['rho[kg/m3]'], bins=20)
axes2[1].hist(testing_features_vector[0]['rho[kg/m3]'], bins=20)

fig4, axes3 = plt.subplots(1,2)
axes3[0].hist(training_features_vector[0]['MVD[mum]'], bins=20)
axes3[1].hist(testing_features_vector[0]['MVD[mum]'], bins=20)

fig5, axes4 = plt.subplots(1,2)
axes4[0].hist(training_features_vector[0]['Mach'], bins=20)
axes4[1].hist(testing_features_vector[0]['Mach'], bins=20)
plt.show()

