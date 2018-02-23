#dataset : http://archive.ics.uci.edu/ml/datasets/banknote+authentication
#Python version 3.5.x

import numpy as np
import random
import collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def modelLogReg(X_original, Y, w, w0, learning_rate, num_iterations):
    #for itr in range(num_iterations):
    cost, itr= 9000, 0

    #X = (X_original - X_original.min(axis=0)) / (X_original.max(axis=0) - X_original.min(axis=0))
    eps = 0.01
    #something like Batch normalization mean = 0, sd = 1

    for i in range(0,X_original.shape[1],1):
        mean = 1/X_original.shape[0] * np.sum(X_original[:,i])
        var  = 1/X_original.shape[0] * np.sum(np.power((X_original[:,i] - mean),2))
        X_original[:, i] = (X_original[:,i] - mean)/np.power((var+eps),0.5)
    
    #X = X_original
    #while num_iterations < abs(cost):
    while itr < num_iterations:

        #w.T . X will give summation of WiXi for a sample X
        #print(w.shape)
        #print(X.shape)
        #print(w0.shape)

        ztemp = np.dot(w.T, X) + w0
        z = np.array(ztemp,dtype=np.float128)
        A = 1/(1+np.exp(-z))

        # print("Shape of A = ",A.shape)
        # compute the gradients and cost
        m = X.shape[1]
        #a = Y * np.log(A)
        #print(A)
        #b = np.log(1 - A)

        cost = (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        w = w + (1/m)*learning_rate * np.dot(X, (Y-A).T)
        w0 = w0 + (1/m)*learning_rate * np.sum(Y-A)
        itr+=1

        if itr % 3000 == 0:
            print("Cost at iteration %i is: %f" % (itr, cost))

    # print("w = ",w)
    # print("b = ",b)
    parameters = {"w": w, "w0": w0}
    return parameters


def classifyLogReg(X, w, w0):

    z = w0 + np.dot(w.T, X)
    A = 1 / (1 + np.exp(-z))

    YPred = np.zeros((1, X.shape[1]))

    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            YPred[0, i] = 1
        else:
            YPred[0, i] = 0

    return YPred


def modelGNB(X, Y):
    cp = {}
    cp = collections.Counter(Y)

    # print(cp[0],cp[1])
    # print(X.shape[0])
    mean = {}
    for i in range(0, X.shape[1], 1):
        mean[i] = {}
        for z in range(0, 2, 1):
            m_sum = 0
            for j in range(0, X.shape[0], 1):
                m_sum += X[j][i] if Y[j] == z else 0

            mean[i][z] = m_sum / cp[z]
            # print(mean[i][z],"i = ",i," z = ",z)

    # print("****")
    variance = {}
    for i in range(0, X.shape[1], 1):
        variance[i] = {}
        for z in range(0, 2, 1):
            var_sum = 0
            for j in range(0, X.shape[0], 1):
                var_sum += (X[j][i] - mean[i][z]) ** 2 if Y[j] == z else 0

            variance[i][z] = var_sum / (cp[z] - 1)
            # print(variance[i][z], "i = ", i, " z = ", z)

    # print("----------------")
    # print(mean.items())
    # print(variance.items())
    return mean, variance, cp


def classifyGNB(X, mean, variance, cp):
    pred = np.zeros((1, X.shape[0]))
    # print(pred.shape)

    for j in range(0, X.shape[0], 1):
        temp_pred = np.zeros((2, 1))

        for z in range(0, 2, 1):
            temp_pred[z] = 1
            for i in range(0, X.shape[1], 1):
                val1 = 1 / (2 * 3.14 * float(variance[i][z])) ** 0.5
                val2 = -(X[j][i] - mean[i][z]) ** 2
                temp_pred[z] *= float(val1 * np.exp(val2 / (2 * variance[i][z])))
                # print(temp_pred[z])

        # print(temp_pred[0],temp_pred[1])
        pred_one_j = float((temp_pred[1] * (cp[1] / (cp[0] + cp[1]))) / (
        temp_pred[1] * (cp[1] / (cp[0] + cp[1])) + temp_pred[0] * (cp[0] / (cp[0] + cp[1]))))
        pred_zero_j = float((temp_pred[0] * (cp[0] / (cp[0] + cp[1]))) / (
        temp_pred[1] * (cp[1] / (cp[0] + cp[1])) + temp_pred[0] * (cp[0] / (cp[0] + cp[1]))))
        pred[0, j] = 1 if pred_one_j > pred_zero_j else 0

    # print( pred.shape)
    return pred



def main():
    #The last column of data is the label.
    data = np.array(np.genfromtxt('./data.csv', delimiter=','))
    cv_fold = 5
    #print(data.shape)
    #print(data[0])

    samples = data.shape[0]

    #This returns a NONE array, so don't assign it to a variable
    np.random.shuffle(data)

    batch_size = (samples/cv_fold)
    learning_rate = 0.1
    num_iterations = 3000
    acc_plotLogReg = {}
    acc_plotGNB = {}

    w = np.zeros((int(data.shape[1]) - 1, 1))
    w0 = 0

    #Initializing the weights here and keep updating them on different folds.

    for i in range(0,cv_fold,1):
        start = batch_size*i
        end = (start + batch_size) if (start + batch_size) < samples else samples
        test_labels  = np.array(data[int(start):int(end),int(data.shape[1])-1])
        test_data = data[int(start):int(end),0:int(data.shape[1])-1]

        train_set = np.append(data[0:int(start),:],data[int(end):samples,:],axis=0)
        #train_data = np.append(data[0:int(start),0:4],data[int(end):samples,0:4],axis=0)

        #Divinding the train_set into fractions for each run of the fold to handle outliers
        #data_fracts = [0.01,0.02,0.5,0.1,0.625,1]
        data_fracts = [1]
        for j in data_fracts:
            #So that the sample size for the fraction does not exceed the max sample available

            sample_fract = j * train_set.shape[0]
            testAccLogReg = 0
            trainAccLogReg = 0
            testAccGNB = 0

            for k in range(0,5,1):
                fstart = random.randint(0,int(train_set.shape[0]-sample_fract))
                fend = fstart+sample_fract

                s_train_data = train_set[int(fstart):int(fend),0:int(data.shape[1])-1]
                s_train_label = np.array(train_set[int(fstart):int(fend), int(data.shape[1])-1])
                # w0 + w.Tx

                #Logistic REGRESSION Prediction and CLassification

                parameters = modelLogReg(s_train_data.T, s_train_label, w, w0, learning_rate, num_iterations)
                w = parameters["w"]
                w0 = parameters["w0"]

                # compute the accuracy for training set and testing set
                train_Pred = classifyLogReg(s_train_data.T, w, w0)
                test_Pred = classifyLogReg(test_data.T, w, w0)
                trainAccLogReg += 100 - (np.sum((np.abs(train_Pred[0,:] - s_train_label))) / train_Pred.shape[1]) * 100
                testAccLogReg += 100 - (np.sum((np.abs(test_Pred[0,:] - test_labels))) / test_Pred.shape[1]) * 100

                '''
                #Gaussian Naive Bayes Prediction and Classification
                param_mean, param_var, class_prob = modelGNB(s_train_data, s_train_label)
                test_Pred = classifyGNB(test_data, param_mean, param_var, class_prob)
                testAccGNB += 100 - (np.sum(np.abs(test_Pred - test_labels)) / test_Pred.shape[1]) * 100
                '''

            acc_plotLogReg[j] = testAccLogReg/5
            acc_plotGNB[j] = testAccGNB / 5
            print("for %f, TestAccLOGReg is %f"%(j,acc_plotLogReg[j]))
            print("for %f, TrainAccLOGReg is %f" % (j, trainAccLogReg/5))
            #print("for %f, testAccGNB is %f" % (j, acc_plotGNB[j]))

        #print("Value : %s" % acc_plot.items())
        '''
        acclstLogReg = sorted(acc_plotLogReg.items())
        acclstGNB = sorted(acc_plotGNB.items())
        xval,yval = zip(*acclstLogReg)
        xval2, yval2 = zip(*acclstGNB)

        plt.title("Gaussian Naive Bayes VS Logistic Regression")
        plt.xlabel("Training Dataset size")
        plt.ylabel("Average Accuracy")
        plt.plot(xval,yval,color="blue",label="Logistic Reg")
        plt.plot(xval2, yval2, color="red", label="Gaussian NB")
        plt.legend(loc='lower right')
        plt.show()

        plt.savefig('fact_vs_accPlot.png')
        '''
        #break
    final_train_data = data[:, 0:int(data.shape[1]) - 1]
    final_train_label = data[: ,int(data.shape[1]) - 1]
    final_train_Pred = classifyLogReg(final_train_data.T, w, w0)

    print(final_train_Pred.shape)
    print(final_train_label.shape)

    acc = 100 - (np.sum((np.abs(final_train_Pred[:,0] - final_train_label))) / final_train_data.shape[0]) * 100
    print("ACCURACY = ",acc)

if __name__ == "__main__":
    main()
