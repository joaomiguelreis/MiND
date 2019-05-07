kNNClassifier builds a multi-label classifier for the airfoil panels (1 classifier, as many outputs as panels on the airfoil)
Cross-validation used as explained in Anabel's email. Figures in kNN_c/
Best is 17NN

kNNRegressor same but with full collection efficiency. FIgures in kNN_r/
Best is 1NN

kNN_1Cell_1Classifier builds a classifier for each panel. cross_val_score is used to evaluate the chosen score
