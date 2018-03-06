'''
Python 3.6.0 :: Anaconda 4.3.0 (64-bit)
Author : Ninad Jadhav(Id : 1213245837)
'''
import numpy as np
from  matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def read_data(filename):
    my_data = np.array(np.genfromtxt(filename,delimiter=','))
    np.random.shuffle(my_data)
    return my_data

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
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X, y)
        prediction_y = neigh.predict(test_X)
        accuracy[k] = np.sum(((test_y == prediction_y)/test_y.shape[0])) * 100
        print(k, accuracy[k])
        
    
if __name__ == "__main__":
    main()
