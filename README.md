# README
This branch contains the scripts used to generate the kNN plots.
Analysis was performed for single/multi-label classification/regression.

kNNClassifer_singleLabel builds one single label classifier for each cell on the airfoil. Binary (impingement/no-impingement) data was used for training and testing. Function cross_val_score was used to evaluate the chosen metrics.
Plots (.eps) are saved to singleLabel/classifier.

kNNRegressor_singleLabel builds one single label regressor for each cell on the airfoil. Full collection efficiency data was used for training. Function cross_val_score was used to evaluate the chosen metrics. The score was computed on the binary condition: predicted values of the collection efficiency  > 0 where converted to 1 (impingement), values < 0 to 0 (no impingement)
Plots (.eps) are saved to singleLabel/regressor.

kNNClassifer_multiLabel builds a single multi-label classifier to predict impingement/no-impingement condition on the whole airfoil. Binary (impingement/no-impingement) data was used for training and testing. Function cross_val_score was NOT used to evaluate the chosen metrics as I could't make it work for multi-label classification/regression. Instead I divided the data in 10 groups and performed the cross validation manually (I followed Anabel's email).
Plots (.eps) are saved to multiLabel/classifier.

kNNRegressor_multiLabel builds a single multi-label regressor to predict impingement/no-impingement condition on the whole airfoil. Binary (impingement/no-impingement) data was used for training. Function cross_val_score was NOT used to evaluate the chosen metrics as I could't make it work for multi-label classification/regression. Instead I divided the data in 10 groups and performed the cross validation manually (I followed Anabel's email). The score is computed on the binary condition (as for kNNRegressor_singleLabel)
Plots (.eps) are saved to multiLabel/regressor.
