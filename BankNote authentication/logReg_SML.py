#dataset : http://archive.ics.uci.edu/ml/datasets/banknote+authentication

import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def model(X, Y, w, w0, learning_rate, num_iterations):
    for itr in range(num_iterations):

        #w.T . X will give summation of WiXi for a sample X
        z = w0 + np.dot(w.T, X)
        A = 1/(1+np.exp(-z))

        # print("Shape of A = ",A.shape)
        # compute the gradients and cost
        m = X.shape[1]
        #cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        w = w + learning_rate * np.dot(X, (Y - A).T)
        w0 = w0 + learning_rate * np.sum(Y - A)

        #if itr % 100 == 0:
            #print("Cost at iteration %i is: %f" % (itr, cost))

    # print("w = ",w)
    # print("b = ",b)
    parameters = {"w": w, "w0": w0}
    return parameters


def classify(X, w, w0):

    z = w0 + np.dot(w.T, X)
    A = 1 / (1 + np.exp(-z))

    YPred = np.zeros((1, X.shape[1]))

    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            YPred[0, i] = 1
        else:
            YPred[0, i] = 0

    return YPred

def main():
    data = np.array(np.genfromtxt('./DataSet/data.csv',delimiter=','))
    cv_fold = 3
    #print(data.shape)
    #print(data[0])

    samples = data.shape[0]
    batch_size = (samples/3)
    learning_rate = 0.1
    num_iterations = 200
    acc_plot={}

    for i in range(0,cv_fold,1):
        start = batch_size*i
        end = (start + batch_size) if (start + batch_size) < samples else samples
        test_labels  = data[int(start):int(end),4]
        test_data = data[int(start):int(end),0:4]

        train_set = np.append(data[0:int(start),:],data[int(end):samples,:],axis=0)
        #train_data = np.append(data[0:int(start),0:4],data[int(end):samples,0:4],axis=0)

        data_fracts = [0.01,0.02,0.05,0.1,0.625,1]
        #data_fracts = [0.01]
        for j in data_fracts:
            #So that the sample size for the fraction does not exceed the max sample available

            sample_fract = j * train_set.shape[0]
            testAcc = 0

            for k in range(0,5,1):
                fstart = random.randint(0,int(train_set.shape[0]-sample_fract))
                fend = fstart+sample_fract

                s_train_data = train_set[int(fstart):int(fend),0:4]
                s_train_label = train_set[int(fstart):int(fend),4]
                # w0 + w.Tx
                w = np.zeros((s_train_data.shape[1], 1))
                w0 = 0
                parameters = model(s_train_data.T, s_train_label, w, w0, learning_rate, num_iterations)
                w = parameters["w"]
                w0 = parameters["w0"]
                # compute the accuracy for training set and testing set

                test_Pred = classify(test_data.T, w, w0)
                testAcc += 100 - (np.sum((np.abs(test_Pred - test_labels))) / test_Pred.shape[1]) * 100

            acc_plot[j] = testAcc/5
            print("for %f, testAcc is %f"%(j,acc_plot[j]))

        #print("Value : %s" % acc_plot.items())
        acclst = sorted(acc_plot.items())
        xval,yval = zip(*acclst)

        plt.plot(xval,yval)
        plt.show()

        plt.savefig('sample_fact_vs_accPlot.png')
        break

if __name__ == "__main__":
    main()
