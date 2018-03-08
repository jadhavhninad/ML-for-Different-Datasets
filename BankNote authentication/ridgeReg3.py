import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from operator import itemgetter

cv_fold=10
data = np.array(np.load('linRegData.npy'))
data = np.insert(data,0,1,axis=1)

X = np.array(data[:,:2])
Y = np.array(data[:,2])
#np.expand_dims(Y, axis = 1)
#print(X.shape)
#print(Y.shape)

for i in range(2,16,1):
	temp = np.power(X[:,1],i)
	X = np.column_stack([X,temp])

random_data = np.column_stack([X,Y])
X_original = np.array(random_data[:,:16])
Y_original = np.array(random_data[:,16])
X = np.array(random_data[:,:16])
Y = np.array(random_data[:,16])

Error_vs_lambda=[]
ls=[0.01,0.05,0.1,0.5,1.0,5,10]
MSE_final ,lval = 0, 0
bsize = X.shape[0] / cv_fold
itr=0

for l in ls: 
		#train1,test1
		lambdav = l
		mse_train, mse_test = 0,0
		for itr in range(0,cv_fold,1):
				start = bsize*itr
				end = start+bsize
				test = X[start:end, :]
				testY = Y[start:end]
				trainX = np.append(X[0:start,:],X[end:100,:],axis=0)
				trainY = np.append(Y[0:start],Y[end:100])
				'''
				b = np.dot(train1.T,train1) + (lambdav*lambdav)*np.identity(16)
				c = np.linalg.inv(b)
				d = np.dot(train1.T,trainY1)
				theta = np.dot(c,d)
				'''
				theta = np.dot(np.linalg.inv(np.dot(trainX.T,trainX) + (lambdav*lambdav)*np.identity(16)), np.dot(trainX.T,trainY))
				#theta = np.dot(np.linalg.inv(np.dot(train1.T,train1) + (lambdav)*np.identity(16)), np.dot(train1.T,trainY1))
				#print(theta)
				train1_new = np.dot(theta,trainX.T)
				test1_new = np.dot(theta,test.T)
				mse_train+=np.mean(np.power((trainY - train1_new),2)/80)
				mse_test+=np.sum(np.power((testY - test1_new),2))
				#print("lambda = ",l,", train1,test1")
				#print("train1 :",mse_train)
				#print("test1 :", mse_test)
				#plt.scatter(Xplt[0:20], testY1, color='blue')
				#plt.plot(Xplt[0:20], test1_new, color='red')
				#plt.show()

		print('final MSE for lambda = ', l , ' : train = ', mse_train/cv_fold , 'test = ', mse_test/cv_fold)
		if MSE_final == 0:
			MSE_final = mse_test
			lval = l
		elif mse_test < MSE_final:
				MSE_final = mse_test
				lval = l

		Error_vs_lambda.append([l,mse_train/cv_fold,mse_test/cv_fold])



print('Lambda value selected as : ' , lval)
theta = np.dot(np.linalg.inv(np.dot(X.T,X) + (lval*lval)*np.identity(16)), np.dot(X.T,Y))
Ynew = np.dot(theta,X.T)
plt.scatter(X[:,1], Y, color='blue')
plt.plot(X[:,1], Ynew, color='red')
plt.show()	
plt.savefig('polynomial_fit.png')
plt.clf()

Error_vs_lambda = np.array(Error_vs_lambda)

plt.xlabel('Train error')
plt.ylabel('Lambda value')
plt.plot(Error_vs_lambda[:,1],Error_vs_lambda[:,0])
plt.savefig('Train_err_vs_lambda.png')
plt.clf()

plt.xlabel('CV Error')
plt.ylabel('Lambda value')
plt.plot(Error_vs_lambda[:,2],Error_vs_lambda[:,0])
plt.savefig('CV_err_vs_lambda.png')
plt.clf()
