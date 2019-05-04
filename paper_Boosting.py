# Import required packages
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, export_graphviz, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

## READING DATA ##

# Setup pandas options
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.5f}'.format

# Get the data
PT_data_1000 = pd.read_excel("../../Data/PTResults-1000_ice.xlsx")

target = PT_data_1000.iloc[:,6:128]
X = PT_data_1000.drop(target, axis=1)  # Just preparing the data
y = target

##################################

## SHUFFLING THE TRAINING/TESTING SPLIT AND GETTING THE BEST TRAINED TREE ##

training_features_vector = []
testing_features_vector = []
training_target_vector = []
testing_target_vector = []

for i in range(1000):
	training_features, testing_features, training_target, testing_target = train_test_split(
		X, y, test_size=0.15, shuffle=True)
	# store
	training_features_vector.append(training_features)
	testing_features_vector.append(testing_features)
	training_target_vector.append(training_target)
	testing_target_vector.append(testing_target)

##################################

## TRAINING THE EXTRA TREE ##

minimum = []
abscissa = []
location = []

for j in range(122):

	ncells_missclass = []

	abscissa.append(j)

	print(j)

	for i in range(10):
		training_features = training_features_vector[i]
		testing_features = testing_features_vector[i]
		training_target = training_target_vector[i]
		testing_target = testing_target_vector[i]

		# Training with Bagging
		boosted = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=10)
		boosted.fit(training_features, training_target)

		# Comparing prediction with testing values
		prediction = boosted.predict(testing_features)

		# Number of missclassified cells
		lst = [item[j] for item in prediction]

		testing_target = np.array(testing_target)
		lst3t = [item[j] for item in testing_target]
		ncells_missclass.append(int(150 * mean_squared_error(lst, lst3t)))

	location.append(np.argmin(ncells_missclass[0:10]))
	minimum.append(min(ncells_missclass))

print(location)

abscissa = 0.428520273131699 + np.array(abscissa)*(1.0-0.428520273131699)/122
minimum = np.array(minimum) / 150 * 100 # %

# Plots
plt.plot(abscissa, minimum)
plt.show()

# Check that the data is equally distributed
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

