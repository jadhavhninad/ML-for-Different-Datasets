from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def min_max_scaling(X):
    X_op = MinMaxScaler().fit_transform(X)
    return X_op;


def read_data(filename):
    my_data = np.array(np.genfromtxt(filename,delimiter=','))
    np.random.shuffle(my_data)
    return my_data

def classify(X,pred_vals,pred_weights):
    y = np.zeros(X.shape[0])
    for (hypo,alpha) in zip(pred_vals,pred_weights):
        y = y + alpha * hypo.predict(X)

    return (np.sign(y))

def adaBoostModel(X,Y,pred_vals,pred_weights,d,X_test,Y_test,itr=10):
    for j in range(itr):
        hypo = DecisionTreeClassifier(max_depth=1)

        hypo.fit(X,Y,sample_weight=d)
        pred_Y = hypo.predict(X)

        #Since there is no dimension, transpose does not matter
        #1. Not normalizing
        eps = np.sum(np.dot(d,(pred_Y != Y)))

        '''
        print((pred_Y != Y)[0:5])
        print(pred_Y[0:5])
        print(eps)
        '''
        alpha = (np.log((1-eps)/eps))/2
        d = d* np.exp(-alpha * Y * pred_Y) # element-wise multiplication
        d = d/ np.sum(d)

        pred_vals.append(hypo)
        pred_weights.append(alpha)

        
        pred_Y_test = classify(X_test,pred_vals,pred_weights)
        acc = (np.sum(pred_Y_test == Y_test))/Y_test.shape[0] * 100
        if j%500 == 0 :
            print(acc, j)
        


def main():
    data = read_data('data.csv')
    X = np.array(min_max_scaling(data[:,0:int(data.shape[1]-1)]))
    Y = np.array(data[:,int(data.shape[1]-1)])

    #Replace class 0 as class -1
    Y[Y<1] = -1
    
    m = int(data.shape[0]/3)
    end = 2*m

    X_train = X[0:end,:]
    X_test = X[end:data.shape[0],:]

    Y_train = Y[0:end]
    Y_test = Y[end:data.shape[0]]

    '''
    print(X[0:5,:])
    print(Y[0:5])
    print(d[0:5])
    print(Y[0:5])
    print(X_test.shape)
    print(Y_test.shape)
    '''
    d = np.ones(X_train.shape[0])/X_train.shape[0]
    pred_vals = []
    pred_weights = []
    adaBoostModel(X_train,Y_train,pred_vals,pred_weights,d,X_test,Y_test,5000)
    #print(pred_vals)
    #print(pred_weights)
    pred_Y_test = classify(X_test,pred_vals,pred_weights)
    pred_Y_train = classify(X_train,pred_vals,pred_weights)

    accTrain = (np.sum(pred_Y_train == Y_train))/Y_train.shape[0] * 100
    acc = (np.sum(pred_Y_test == Y_test))/Y_test.shape[0] * 100
    print(accTrain,acc)

 

if __name__ == "__main__":
    main()
        
