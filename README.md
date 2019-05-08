kNNRegressor builds a single, multi-label regressor to predict collection efficiency on airfoil. Collection efficiency data is then converted to ice/no ice condition on all panels of the airfoil. Performances are evaluated on binary condition. Performances are evaluated with cross validation. Function cross_val_score was not used as it only supports single label classifiers (I think)
Score can be selected by uncommenting the wanted SCORE variable. SCORE=error is wrong predictions/all test samples (i.e. 1-accuracy). Plots (eps format) are saved to kNN_r/

kNNClassifier builds a single, multi-label classifier to predict ice/no-ice condition.
Performances are evaluated with cross validation. Function cross_val_score was not used as it only supports single label classifiers (I think)
Score can be selected by uncommenting the wanted SCORE variable. SCORE=error is wrong predictions/all test samples (i.e. 1-accuracy). Plots (eps format) are saved to kNN_c/
