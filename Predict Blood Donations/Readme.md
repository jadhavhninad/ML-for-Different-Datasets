## [](header-2)Logistic Regression
Without using feature scaling and no regularization, computation error was seen ('overflow error'). Achieved accuracy on test set was ~75%. 
Feature scaling resolved the error but no marked improvement was seen in the accuracy. 

Using a 5-fold CV method and 3000 iterations, train accuracy reached about 78%, while 80% accuracy was seen for test data. Increasing the 
numberof iterations did not increase accuracy. Same for 10-fold CV.

Using n-fold CV gave an accuracy of 76% on the entire dataset


-> Normalizing using mean and variance gives the same accuracy.


## [](header-2) Boosting
The accuracy improved to 79.6% (For 5000 iterations). On train data, the accuracy is about 83%. The 1R (decision tree stump) library from python scitlearn was used. Need to do some more tests on this to check if accuracy can be improved further or not.

## [](header-2) K- Nearest Neighbour 
Using KNN approach (default package), building from scratch gives accuracy of 77% on an average.
