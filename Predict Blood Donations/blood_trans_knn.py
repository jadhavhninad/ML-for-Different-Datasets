'''
Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
Author : Ninad Jadhav(Id : 1213245837)
'''
import numpy as np
from  matplotlib import pyplot as plt
import heapq
import operator

def read_data(filename):
    my_data = np.array(np.genfromtxt(filename,delimiter=','))
    np.random.shuffle(my_data)
    return my_data

def euclidean_distance(a,b):
    distance = np.sum(np.power((np.subtract(a,b)),2))
    return distance

def knn_model(X,y,test_X,k):
    pred_y=np.array(np.zeros(test_X.shape[0]))
    for test_sample in range (0,test_X.shape[0],1):
        vote=[]
        for train_sample in range (0,X.shape[0],1):
            dist = euclidean_distance(X[train_sample,:],test_X[test_sample,:])
            vote.append((dist, y[train_sample]))

        k_nearest = heapq.nsmallest(k,vote,key=lambda x:x[0])
        digit_pred={0.0:0, 1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0}

        for neighbours,label in k_nearest:
            digit_pred[label] += 1

        pred_y[test_sample] = max(digit_pred.items(), key=operator.itemgetter(1))[0]
        
    return pred_y


def main():
    data = read_data('data.csv')
    #np.random.shuffle(data)
    X_og = np.array(data[:,0:int(data.shape[1]-1)])
    Y_og = np.array(data[:,int(data.shape[1]-1)])

    #Replace class 0 as class -1
    #Y[Y<1] = -1
    
    m = int(data.shape[0]/3)
    end = 2*m

    X = X_og[0:end,:]
    y = Y_og[0:end]
    
    test_X = X_og[end:data.shape[0],:]
    test_y = Y_og[end:data.shape[0]]

    '''
    print(X[0,:])
    print(y[0])
    print(test_X[0,])
    print(test_y[0])
    '''
    accuracy={}
    kvals = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    #kvals = [1]
    for k in kvals:
        prediction_y = knn_model(X,y,test_X,k)
        accuracy[k] = np.sum(((test_y == prediction_y)/test_y.shape[0])) * 100
        print(k, accuracy[k])
        
    
if __name__ == "__main__":
    main()
