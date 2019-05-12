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
----
# Results
Single and multi-label classification/regression yield the same results. In the following are presented only the results of the single label classification/regression

## Classification (kNNClassifer_singleLabel)
Different values for k (number of neighbours) were tested. The integral precision and recall scores were used to asses the best value for k. The integral score is defined as:
![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bintegral%20score%7D%20%3D%20%5Cint_%7B0%7D%5E%7B1%7D%5Ctextrm%7Bscore%7D%28s%29%20ds)
### Precision
In the following image the precision score (%) is plotted on the surface of the airfoil:
![Alt text](.readme/class_precision.png?raw=true "Title")

In the following image the integral precision score (%) is plotted against the number of neighbours. k=1 yields the highest integral precision (21):
![Alt text](.readme/int_class_precision.png?raw=true "Title")

### Recall
In the following image the recall score (%) is plotted on the surface of the airfoil:
![Alt text](.readme/class_recall.png?raw=true "Title")

In the following image the integral recall score (%) is plotted against the number of neighbours. k=1 yields the highest integral recall (29):
![Alt text](.readme/int_class_recall.png?raw=true "Title")

## Regression (kNNRegressor_singleLabel)
Different values for k (number of neighbours) were tested. The integral precision and recall scores were used to asses the best value for k.
### Precision
In the following image the precision score (%) is plotted on the surface of the airfoil:
![Alt text](.readme/reg_precision.png?raw=true "Title")

In the following image the integral precision score (%) is plotted against the number of neighbours. k=1 yields the highest integral precision (29):
![Alt text](.readme/int_reg_precision.png?raw=true "Title")

### Recall
In the following image the recall score (%) is plotted on the surface of the airfoil:
![Alt text](.readme/reg_recall.png?raw=true "Title")

In the following image the integral recall score (%) is plotted against the number of neighbours. k>45 yields the highest integral recall (53):
![Alt text](.readme/int_reg_recall.png?raw=true "Title")
