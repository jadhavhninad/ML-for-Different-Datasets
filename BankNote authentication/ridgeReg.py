import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from operator import itemgetter

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

np.random.shuffle(random_data)

X = np.array(random_data[:,:16])
Y = np.array(random_data[:,16])

'''
temp = np.power(Xplt,2)
X = np.column_stack([X,temp])
'''
#print(X.shape)
#A_NEW = A[Start_index : stop_index, start_index : stop_index)]

test1 = X[0:20,:]
test2 = X[20:40,:]
test3 = X[40:60,:]
test4 = X[60:80,:]
test5 = X[80:100,:]

testY1 = Y[0:20]
testY2 = Y[20:40]
testY3 = Y[40:60]
testY4 = Y[60:80]
testY5 = Y[80:100]


train1 = X[20:100,:]
#print(trainY1.shape)
train2 = np.append(X[0:20,:],X[40:100],axis=0)
train3 = np.append(X[0:40,:],X[60:100],axis=0)
train4 = np.append(X[0:60,:],X[80:100],axis=0)
train5 = X[0:80,:] 

trainY1 = np.array(Y[20:100])
trainY2 = np.append(Y[0:20],Y[40:100])
trainY3 = np.append(Y[0:20],Y[40:100])
trainY4 = np.append(Y[0:20],Y[40:100])
trainY5 = np.array(Y[0:80])


plt.scatter(X[0:20,1], testY1, color='blue')
plt.scatter(X[20:40,1], testY2, color='green')
plt.scatter(X[40:60,1], testY3, color='yellow')
plt.scatter(X[60:80,1], testY4, color='orange')
plt.scatter(X[80:100,1], testY5, color='pink')
plt.savefig('datadistribut.png')
plt.clf()

Error_vs_lambda=[]
ls=[0.01,0.05,0.1,0.5,1.0,5,10]
MSE_final ,lval = 10000, 0


for l in ls: 
	#train1,test1
	lambdav = l
	mse_train, mse_test = 0,0
	'''
	b = np.dot(train1.T,train1) + (lambdav*lambdav)*np.identity(16)
	c = np.linalg.inv(b)
	d = np.dot(train1.T,trainY1)
	theta = np.dot(c,d)
	'''
	theta = np.dot(np.linalg.inv(np.dot(train1.T,train1) + (lambdav*lambdav)*np.identity(16)), np.dot(train1.T,trainY1))
	#theta = np.dot(np.linalg.inv(np.dot(train1.T,train1) + (lambdav)*np.identity(16)), np.dot(train1.T,trainY1))
	#print(theta)
	train1_new = np.dot(theta,train1.T)
	test1_new = np.dot(theta,test1.T)
	mse_train+=np.mean(np.power((trainY1 - train1_new),2)/80)
	mse_test+=np.sum(np.power((testY1 - test1_new),2))
	#print("lambda = ",l,", train1,test1")
	#print("train1 :",mse_train)
	#print("test1 :", mse_test)
        #plt.scatter(Xplt[0:20], testY1, color='blue')
	#plt.plot(Xplt[0:20], test1_new, color='red')
	#plt.show()

	#train2,test2
	theta = np.dot(np.linalg.inv(np.dot(train2.T,train2) + (lambdav*lambdav)*np.identity(16)), np.dot(train2.T,trainY2))
	#theta = np.dot(np.linalg.inv(np.dot(train2.T,train2) + (lambdav)*np.identity(16)), np.dot(train2.T,trainY2))
	#print(theta)
	train2_new = np.dot(theta,train2.T)
	test2_new = np.dot(theta,test2.T)
	mse_train+=np.mean(np.power((trainY2 - train2_new),2)/80)
	mse_test+=np.sum(np.power((testY2 - test2_new),2))
	#print("lambda = ",l,", train1,test1")
	#print("train2 :",mse_train)
	#print("test2 :", mse_test)
	#plt.scatter(Xplt[20:40], testY2, color='green')
	#plt.plot(Xplt[20:40], test2_new, color='red')
	#plt.show()

	#train3,test3
	theta = np.dot(np.linalg.inv(np.dot(train3.T,train3) + (lambdav*lambdav)*np.identity(16)), np.dot(train3.T,trainY3))
	#theta = np.dot(np.linalg.inv(np.dot(train3.T,train3) + (lambdav)*np.identity(16)), np.dot(train3.T,trainY3))
	#print(theta)
	train3_new = np.dot(theta,train3.T)
	test3_new = np.dot(theta,test3.T)
	mse_train+=np.mean(np.power((trainY3 - train3_new),2)/80)
	mse_test+=np.sum(np.power((testY3 - test3_new),2))
	#print("lambda = ",l,", train1,test1")
	#print("train3 :",mse_train)
	#print("test3 :", mse_test)
	#plt.scatter(Xplt[40:60], testY3, color='yellow')
	#plt.plot(Xplt[40:60], test3_new, color='red')
	#plt.show()

	#train4,test4
	theta = np.dot(np.linalg.inv(np.dot(train4.T,train4) + (lambdav*lambdav)*np.identity(16)), np.dot(train4.T,trainY4))
	#theta = np.dot(np.linalg.inv(np.dot(train4.T,train4) + (lambdav)*np.identity(16)), np.dot(train4.T,trainY4))
	#print(theta)
	train4_new = np.dot(theta,train4.T)
	test4_new = np.dot(theta,test4.T)
	mse_train+=np.mean(np.power((trainY4 - train4_new),2)/80)
	mse_test+=np.sum(np.power((testY4 - test4_new),2))
	#print("lambda = ",l,", train1,test1")
	#print("train4 :",mse_train)
	#print("test4 :", mse_test)
	#plt.scatter(Xplt[60:80], testY4, color='orange')
	#plt.plot(Xplt[60:80], test4_new, color='red')
	#plt.show()

	#train2,test2
	theta = np.dot(np.linalg.inv(np.dot(train5.T,train5) + (lambdav*lambdav)*np.identity(16)), np.dot(train5.T,trainY5))
	train5_new = np.dot(theta,train5.T)
	test5_new = np.dot(theta,test5.T)
	mse_train+=np.mean(np.power((trainY5 - train5_new),2)/80)
	mse_test+=np.sum(np.power((testY5 - test5_new),2))
	
	print('final MSE for lambda = ', l , ' : train = ', mse_train/5 , 'test = ', mse_test/5)
	if mse_test < MSE_final:
		MSE_final = mse_test
		lval = l

	Error_vs_lambda.append([l,mse_train/5,mse_test/5])
	
print('Lambda value selected as : ' , lval)
theta = np.dot(np.linalg.inv(np.dot(X.T,X) + (lval*lval)*np.identity(16)), np.dot(X.T,Y))
Ynew = np.dot(theta,X_original.T)
plt.scatter(X_original[:,1], Y_original, color='blue')
plt.plot(X_original[:,1], Ynew, color='red')
plt.show()	
plt.savefig('polynomial_fit.png')
plt.clf()

Error_vs_lambda = np.array(Error_vs_lambda)

plt.xlabel('Train error')
plt.ylabel('Lambda value')
plt.plot(Error_vs_lambda[:,0],Error_vs_lambda[:,1])
plt.savefig('Train_err_vs_lambda.png')
plt.clf()

plt.xlabel('CV Error')
plt.ylabel('Lambda value')
plt.plot(Error_vs_lambda[:,0],Error_vs_lambda[:,2])
plt.savefig('CV_err_vs_lambda.png')
plt.clf()


