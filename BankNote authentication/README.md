## [](header-2)A logistic regression model for banknote Authentication data:

[Link to UCR dataset](http://archive.ics.uci.edu/ml/datasets/banknote+authentication)


Implementing 3 fold cross validation with different size of datasets

## [](header-3)Using Higher Order Polynomial curve fitting
Using ridge regression for higher order polynomial curve fit. Optimal learning rate is selected using cross validation

run: `python ridgeReg3.py`

Note: Due to linearity in given data (data is sorted along X-axis), using a 5 fold cv error gives bad lambda value. This can be mitigated by either shuffling the data(though the output will depend how data gets shuffled internally) or using a higher order CV fold.

Using 10-fold CV gives lambda = 0.01 and using 100-fold CV give a value of 0.5(this fit is much smoother and seems to be the ideal value)

