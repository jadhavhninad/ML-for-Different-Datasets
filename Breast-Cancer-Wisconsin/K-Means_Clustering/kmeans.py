import numpy as np
import random as rd
import matplotlib.pyplot as plt

class kmeans_model:

    def __init__(self,X,k,itr):
        self.X = X
        self.k = k
        self.means = []
        self.itr = itr
        self.L = 0

    def init_means(self):
        #Using K_means++ strategy
        mean = rd.randint(0,len(self.X))
        self.means.append(self.X[mean,:])
        dist_to_mean = [10000] * len(self.X)

        #Calculate the distance for all points from their closest mean.
        #At end of each iteration, return the index of max dist_to_mean and assign it as new mean

        for i in range(self.k - 1):
            for point in range(len(self.X)):
                temp = [np.linalg.norm(self.X[point, :] - m) for m in self.means]
                dist_to_mean[point] = min(temp)

                '''
                for m in self.means:
                    temp = np.linalg.norm((self.X[point,:] - self.X[m,:]))
                    #print("temp = ", temp)
                    if temp < dist_to_mean[point]:
                        dist_to_mean[point] = temp
                #print("------")
                '''

            self.means.append(self.X[np.argmax(dist_to_mean),:])

        #print("new means = ", self.means)


    def getPotential(self,clusters):
        # Calculate potential function value
        for center in range(len(clusters)):
            dist = 0
            for i in range(len(clusters[center])):
                dist += np.sum(np.linalg.norm(self.X[i, :] - self.means[center]))

            self.L = self.L + dist

        return self.L


    def cost(self):
        center_dist_old = 0
        for i in range(self.itr):
            clusters = {}
            for i in range(self.k):
                clusters[i] = []

            #======Assign all data points to a cluster===========
            for point in range(len(self.X)):
                dist_to_center = [np.linalg.norm(self.X[point,:] - m) for m in self.means]
                point_class = dist_to_center.index(min(dist_to_center))
                clusters[point_class].append(point)


            #=====Recalculate the cluster centers==============

            old_centers = list(self.means)
            # average the cluster datapoints to re-calculate the centers

            for center in range(len(clusters)):
                cluster_X = []
                for i in range(len(clusters[center])):
                    cluster_X.append(self.X[i,:])

                #print(cluster_X)
                self.means[center] = np.average(cluster_X, axis=0)


            #Get largest Cluster
            cl = 0
            largest_cluster = -1
            for center in range(len(clusters)):
                if len(clusters[center]) > cl:
                    cl = len(clusters[center])
                    largest_cluster = center

            # Check if any cluster is empty and if yes, split largest cluster
            for center in range(len(clusters)):
                if len(clusters[center]) == 0:
                    clusters[center] = list(clusters[largest_cluster][0:cl/2])
                    clusters[largest_cluster] = clusters[largest_cluster][cl/2 + 1 : cl]

                    #Update the centers for them as some random point
                    self.means[center] = self.X[rd.choice(clusters[center]),:]
                    self.means[largest_cluster] = self.X[rd.choice(clusters[largest_cluster]), :]

                    #print("Post handling the 0 size")
                    #for center in range(len(clusters)):
                        #print(len(clusters[center]))

                    break


            # =======If optimal centrs already found, then stop the algorithm======
            center_dist = 0
            for i in range(self.k):
                #print(np.sum(np.linalg.norm(old_centers[i] - self.means[i])))
                center_dist += np.sum(np.linalg.norm(old_centers[i] - self.means[i]))

            #print("--------",center_dist)

            if center_dist < self.k*0.001 or (center_dist_old - center_dist == 0):

                #Calculate potential function value
                self.L = self.getPotential(clusters)
                break;

            center_dist_old = center_dist

        if self.L == 0:
            self.L = self.getPotential(clusters)

        return self.L




def main():

    data = np.genfromtxt('./data.csv', delimiter=',')
    X = data[:,1:10]
    #print(X.shape)
    k_vals=[2,3,4,5,6,7,8]
    L=[]
    itr=500

    for k in k_vals:
        model = kmeans_model(X,k,itr)
        model.init_means()
        L_val = model.cost()
        L.append(L_val)
        print("=========done=========")

    print(L)
    #Plot the L and k_vals
    plt.plot(k_vals,L)
    plt.savefig('plot_val')

if __name__ == "__main__":
    main()