import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


#Defining class object for KD Tree
class KDTree:
    #Define a base initializer
    def __init__(self, matrix):
        self.feature_means = np.mean(matrix,axis=0) #Rows are samples, columns are features
        self.max_depth = 0
        self.global_matrix = matrix
        self.location = 'root'
        self.depth = 0

        remainder = np.shape(matrix)[0]
        while remainder > 0 :
            self.max_depth = self.max_depth + 1
            remainder = int(remainder/4)

        print('The maximum depth of this tree is:', self.max_depth)

    #Define a traversal
    def traverse_tree(self,vector):

        self.local_matrix = self.global_matrix
        self.temp_matrix = self.global_matrix
        self.vector = vector

        for depth in range(self.max_depth):
            #Classify vector first - changes level
            self.classify()
            #Update feature mean to this particular class
            self.update_local_matrix()
            # Check for early exit
            if self.local_matrix.size == 0 or depth == self.max_depth - 2:
                break
            else:
                self.feature_means = np.mean(self.local_matrix, axis=0)
                self.temp_matrix = self.local_matrix

            #Increase depth
            self.depth = self.depth + 1

        nearest_neighbor = self.nearest_neighbor_classify()

        #print('The nearest neighbor is:',nearest_neighbor[0:2])
        #print('The classification is:', nearest_neighbor[2])

        return nearest_neighbor


    #Find nearest neighbor from array and classify
    def nearest_neighbor_classify(self):

        idx = np.abs(self.temp_matrix[:,0:2] - np.reshape(self.vector,(1,2)))
        idx = np.sqrt(idx[:,0]**2 + idx[:,1]**2).argmin()
        return self.temp_matrix[idx,:]


    #Define a classifier
    def classify(self):
        if self.vector[0] > self.feature_means[0] and self.vector[1] > self.feature_means[1]:
            self.location = 'top-right'

        elif self.vector[0] > self.feature_means[0] and self.vector[1] < self.feature_means[1]:
            self.location = 'bottom-right'

        elif self.vector[0] < self.feature_means[0] and self.vector[1] < self.feature_means[1]:
            self.location = 'bottom-left'

        else:
            self.location = 'top-left'


    #Update local matrix (i.e. data in same class as current class of vector)
    def update_local_matrix(self):
        if self.location == 'top-right':

            self.local_matrix = self.local_matrix[np.where(
                (self.local_matrix[:, 0] > self.feature_means[0]) & (self.local_matrix[:, 1] > self.feature_means[1]))]

        elif self.location == 'bottom-right':

            self.local_matrix = self.local_matrix[np.where(
                (self.local_matrix[:, 0] > self.feature_means[0]) & (self.local_matrix[:, 1] < self.feature_means[1]))]

        elif self.location == 'bottom-left':

            self.local_matrix = self.local_matrix[np.where(
                (self.local_matrix[:, 0] < self.feature_means[0]) & (self.local_matrix[:, 1] < self.feature_means[1]))]

        else:

            self.local_matrix = self.local_matrix[np.where(
                (self.local_matrix[:, 0] < self.feature_means[0]) & (self.local_matrix[:, 1] > self.feature_means[1]))]



if __name__ == "__main__":
    #Set seed for reproducibility
    np.random.seed(10)

    #Generate some clustered data
    num_clusters = 10
    total_data = 10000
    covariance = [[2, 1], [1, 2]]

    #Labeling zeroes
    for i in range(num_clusters//2):
        mean = [np.random.uniform(low=0,high=6), np.random.uniform(low=1,high=10)]
        cluster_data_temp = np.random.multivariate_normal(mean, covariance, total_data // num_clusters)
        labels = np.zeros(shape=(total_data//num_clusters, 1))

        if i == 0:
            cluster_data_1 = np.hstack((cluster_data_temp, labels))
        else:
            cluster_data_temp = np.hstack((cluster_data_temp, labels))
            cluster_data_1 = np.concatenate((cluster_data_1,cluster_data_temp),axis=0)

    # Labeling ones
    for i in range(num_clusters // 2):
        mean = [np.random.uniform(low=0, high=6), np.random.uniform(low=1, high=10)]
        cluster_data_temp = np.random.multivariate_normal(mean, covariance, total_data // num_clusters)
        labels = np.ones(shape=(total_data // num_clusters, 1))

        if i == 0:
            cluster_data_2 = np.hstack((cluster_data_temp, labels))
        else:
            cluster_data_temp = np.hstack((cluster_data_temp, labels))
            cluster_data_2 = np.concatenate((cluster_data_2, cluster_data_temp), axis=0)

    #Concatenate matrix and randomize rows
    matrix_data = np.concatenate((cluster_data_1,cluster_data_2),axis=0)
    np.random.shuffle(matrix_data)

    training_data = matrix_data[0:total_data//2, :]
    testing_data = matrix_data[total_data//2:, :]

    # group = matrix_data[:, 2]
    # cdict = {0: 'blue', 1: 'red'}
    # #
    # fig, ax = plt.subplots()
    # for g in np.unique(group):
    #     ix = np.where(group == g)
    #     ax.scatter(matrix_data[:,0][ix], matrix_data[:,1][ix], c=cdict[g], label=g, s=10)
    #
    # ax.set_xlabel('$X$')
    # ax.set_ylabel('$y$')
    # ax.legend()
    #
    # plt.show()
    # exit()

    #Plotting
    scatter_x = matrix_data[:,0]
    scatter_y = matrix_data[:,1]
    group = matrix_data[:,2]
    cdict = {0: 'blue', 1: 'red'}
    #
    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=10)
    #

    #Checking performance of classifier on testing data
    tree_object = KDTree(training_data)
    incorrect = 0
    for i in range(np.shape(testing_data)[0]):
        vector = testing_data[i,0:2]
        classification = int(tree_object.traverse_tree(vector)[2])

        if classification == 1 and int(testing_data[i,2]) == 1:
            ax.scatter(vector[0], vector[1], c='black', marker='o', s=10,label='Correct class of 1')
        elif classification == 1 and int(testing_data[i,2]) == 0:
            ax.scatter(vector[0], vector[1], c='green', marker='o', s=10, label='Incorrect class of 1')
        elif classification == 0 and int(testing_data[i,2]) == 0:
            ax.scatter(vector[0], vector[1], c='brown', marker='o', s=10, label='Correct class of 0')
        elif classification == 0 and int(testing_data[i,2]) == 1:
            ax.scatter(vector[0], vector[1], c='violet', marker='o', s=10, label='Incorrect class of 0')

        incorrect = incorrect + np.abs(classification-testing_data[i,2])

    print ('Classification accuracy:',float(1.0-incorrect/5000))

    #Complete plotting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),loc='lower left')
    plt.show()