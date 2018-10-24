import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def linear_classifier_function(matrix):
    temp = np.dot(np.transpose(matrix[:,0:2]),matrix[:,0:2])
    temp = np.linalg.inv(temp)
    temp = np.dot(temp,np.transpose(matrix[:,0:2]))
    beta = np.dot(temp,matrix[:,2])

    return beta

def test_classifier(vector,beta):
    val = np.dot(vector,beta)

    if val > 0.5:
        return 1
    else:
        return 0

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

    # Plotting
    scatter_x = matrix_data[:, 0]
    scatter_y = matrix_data[:, 1]
    group = matrix_data[:, 2]
    cdict = {0: 'blue', 1: 'red'}
    #
    fig, ax = plt.subplots()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=10)
    #
    cdict_class = {0: 'black', 1: 'yellow'}

    # Checking performance of classifier on testing data
    beta = linear_classifier_function(training_data)
    incorrect = 0
    for i in range(np.shape(testing_data)[0]):
        vector = testing_data[i, 0:2]
        classification = test_classifier(vector,beta)
        incorrect = incorrect + np.abs(classification - testing_data[i, 2])

        if classification == 1 and int(testing_data[i,2]) == 1:
            ax.scatter(vector[0], vector[1], c='black', marker='o', s=10,label='Correct class of 1')
        elif classification == 1 and int(testing_data[i,2]) == 0:
            ax.scatter(vector[0], vector[1], c='green', marker='o', s=10, label='Incorrect class of 1')
        elif classification == 0 and int(testing_data[i,2]) == 0:
            ax.scatter(vector[0], vector[1], c='brown', marker='o', s=10, label='Correct class of 0')
        elif classification == 0 and int(testing_data[i,2]) == 1:
            ax.scatter(vector[0], vector[1], c='violet', marker='o', s=10, label='Incorrect class of 0')

    print('Classification accuracy:', float(1.0 - incorrect / 5000))

    # Complete plotting
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),loc='lower left')
    plt.show()