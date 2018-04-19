import numpy as np

def main():
    lst=[0,1,2,3,4,5,6,7,8,9]
    y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]

    print("----------------ITR 1---------theta = 2.5------")

    pred=[1,1,1,-1,-1,-1,-1,-1,-1,-1]

    #itr1:
    D = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    misSum=0
    for i in range(10):
        misSum += D[i] if y[i] != pred[i] else 0

    print("misSum = ", misSum)
    esp = 1/np.sum(D) * misSum
    print("esp = ", esp)
    alpha = 0.5*(np.log((1-esp)/esp))
    print("alpha = ", alpha)

    for i in range(10):
        D[i] = D[i] * np.exp(-alpha*y[i]*pred[i])

    #print("D b4 normalization " , D)
    Z = np.sum(D)
    print("Z = ",Z)
    D = D/Z
    print(D)

    wtErr=0
    for i in range(10):
        wtErr+= D[i] if y[i] != pred[i] else 0

    print("err = ", wtErr)

    print("----------------ITR 2---------theta = 8.5------")
    #theta = 8.5
    pred=[1,1,1,1,1,1,1,1,1,-1]

    #itr1:
    misSum=0
    for i in range(10):
        misSum += D[i] if y[i] != pred[i] else 0

    print("misSum = ", misSum)
    esp = 1/np.sum(D) * misSum
    print("esp = ", esp)
    alpha = 0.5*(np.log((1-esp)/esp))
    print("alpha = ", alpha)

    for i in range(10):
        D[i] = D[i] * np.exp(-alpha*y[i]*pred[i])

    print("D b4 normalization " , D)
    Z = np.sum(D)
    print("Z = ",Z)
    D = D/Z
    print(D)

    wtErr = 0
    for i in range(10):
        wtErr += D[i] if y[i] != pred[i] else 0

    print("err = ", wtErr)

    print("----------------ITR 3---------theta = 5.5------")
    #theta =5.5
    #Since the accuracy is less than 50%, we flip the classifier prediction
    pred = [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1]

    #itr1:
    misSum=0
    for i in range(10):
        misSum += D[i] if y[i] != pred[i] else 0

    print("misSum = ", misSum)
    esp = 1/np.sum(D) * misSum
    print("esp = ", esp)
    alpha = 0.5*(np.log((1-esp)/esp))
    print("alpha = ", alpha)

    for i in range(10):
        D[i] = D[i] * np.exp(-alpha*y[i]*pred[i])

    #print("D b4 normalization " , D)
    Z = np.sum(D)
    print("Z = ",Z)
    D = D/Z
    print(D)

    wtErr = 0
    for i in range(10):
        wtErr += D[i] if y[i] != pred[i] else 0

    print("err = ", wtErr)

    print("----------------ITR 4---------theta = 0------")
    # theta =5.5
    # Since the accuracy is less than 50%, we flip the classifier prediction
    pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # itr1:
    misSum = 0
    for i in range(10):
        misSum += D[i] if y[i] != pred[i] else 0

    print("misSum = ", misSum)
    esp = 1 / np.sum(D) * misSum
    print("esp = ", esp)
    alpha = 0.5 * (np.log((1 - esp) / esp))
    print("alpha = ", alpha)

    for i in range(10):
        D[i] = D[i] * np.exp(-alpha * y[i] * pred[i])

    #print("D b4 normalization ", D)
    Z = np.sum(D)
    print("Z = ", Z)
    D = D / Z
    print(D)

    wtErr = 0
    for i in range(10):
        wtErr += D[i] if y[i] != pred[i] else 0

    print("err = ", wtErr)


if __name__ == "__main__":
    main()