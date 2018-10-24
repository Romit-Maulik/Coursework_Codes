import numpy as np
import matplotlib.pyplot as plt
import gzip as gz


def load_data():

    np.random.seed(2)

    #Training images
    data_file = gz.open(r'train-images-idx3-ubyte.gz','rb')
    training_images = data_file.read()
    data_file.close()
    training_images = bytearray(training_images)[16:]
    training_images = np.reshape(np.asarray(training_images),(60000,784))

    #Training labels
    data_file = gz.open(r'train-labels-idx1-ubyte.gz', 'rb')
    training_labels = data_file.read()
    data_file.close()
    training_labels = bytearray(training_labels)[8:]
    training_labels = np.reshape(np.asarray(training_labels),(60000,1))

    # Testing images
    data_file = gz.open(r't10k-images-idx3-ubyte.gz', 'rb')
    testing_images = data_file.read()
    data_file.close()
    testing_images = bytearray(testing_images)[16:]
    testing_images = np.reshape(np.asarray(testing_images), (10000, 784))

    # Testing labels
    data_file = gz.open(r't10k-labels-idx1-ubyte.gz', 'rb')
    testing_labels = data_file.read()
    data_file.close()
    testing_labels = bytearray(testing_labels)[8:]
    testing_labels = np.reshape(np.asarray(testing_labels),(10000,1))


    #Creating training data
    idx1 = np.ndarray.flatten(np.asarray((np.where(training_labels[:, 0] == 1))))
    idx2 = np.ndarray.flatten(np.asarray((np.where(training_labels[:, 0] == 2))))
    idx7 = np.ndarray.flatten(np.asarray((np.where(training_labels[:, 0] == 7))))
    idx1 = np.random.choice(idx1, 200, replace=False)
    idx2 = np.random.choice(idx2, 200, replace=False)
    idx7 = np.random.choice(idx7, 200, replace=False)

    idx = np.concatenate((idx1, idx2, idx7), axis=0)
    training_images = training_images[idx, :]
    training_labels = training_labels[idx, :]

    #Creating testing data
    idx1 = np.ndarray.flatten(np.asarray((np.where(testing_labels[:, 0] == 1))))
    idx2 = np.ndarray.flatten(np.asarray((np.where(testing_labels[:, 0] == 2))))
    idx7 = np.ndarray.flatten(np.asarray((np.where(testing_labels[:, 0] == 7))))
    idx1 = np.random.choice(idx1, 50, replace=False)
    idx2 = np.random.choice(idx2, 50, replace=False)
    idx7 = np.random.choice(idx7, 50, replace=False)

    idx = np.concatenate((idx1, idx2, idx7), axis=0)
    testing_images = testing_images[idx, :]
    testing_labels = testing_labels[idx, :]

    return training_images, training_labels, testing_images, testing_labels

def distance_metric(sample,data):
    '''
    :param sample: An image feature vector to be classified
    :param data:  Training data of many feature vectors
    :return; set of distances
    '''
    ret_mat = np.zeros(shape=(np.shape(data)[0],),dtype='double')
    for i in range(np.shape(data)[0]):
        training_data = data[i,:]
        ret_mat[i] = np.sum(np.square(1.0*(sample-training_data)))

    return ret_mat

def classify_brute_force(training_images,training_labels,testing_images, testing_labels,k):
    '''
    :param testing_images: Image feature vectors to be classified
    :param training_images: Training data of many feature vectors
    :param training_labels: Training labels of many feature vectors
    :param testing_labels: Testing labels of many feature vectors for accuracy assessment
    :param k: Number of nearest neighbors
    :return:
    '''

    correct = 0
    for i in range(np.shape(testing_images)[0]):
        sample = testing_images[i,:]

        row_id = distance_metric(sample,training_images).argsort()[0:k]
        pred_label = training_labels[row_id,0]

        classification = np.argmax([np.sum(pred_label==1), np.sum(pred_label==2), np.sum(pred_label==7)])

        if classification == 0:
            prediction = 1
        elif classification == 1:
            prediction = 2
        else:
            prediction = 7

        true_label = testing_labels[i,0]

        if prediction == true_label:
            correct = correct + 1

    #print('Classification accuracy: ',correct/np.shape(testing_images)[0])

    return correct/np.shape(testing_images)[0]

def classify_brute_force_test(training_images,training_labels,testing_images, testing_labels,k,disp):
    '''
    :param testing_images: Image feature vectors to be classified
    :param training_images: Training data of many feature vectors
    :param training_labels: Training labels of many feature vectors
    :param testing_labels: Testing labels of many feature vectors for accuracy assessment
    :param k: Number of nearest neighbors
    :return:
    '''

    correct = 0
    for i in range(np.shape(testing_images)[0]):
        sample = testing_images[i,:]
        row_id = distance_metric(sample,training_images).argsort(axis=0)[0:k]

        pred_label = training_labels[row_id[0],0]
        classification = np.argmax([np.sum(pred_label==1), np.sum(pred_label==2), np.sum(pred_label==7)])

        if classification == 0:
            prediction = 1
        elif classification == 1:
            prediction = 2
        else:
            prediction = 7

        true_label = testing_labels[i,0]

        if prediction == true_label:
            correct = correct + 1
        elif disp == True:
            # Visualize
            check_row_nn = np.reshape(training_images[row_id[0],:],(28,28))
            check_row_test = np.reshape(testing_images[i, :], (28, 28))

            plt.figure()
            plt.imshow(check_row_test)
            plt.title('Test image')
            plt.show()

            plt.figure()
            plt.imshow(check_row_nn)
            plt.title('Nearest Neighbor')
            plt.show()


    print('Classification accuracy: ',correct/np.shape(testing_images)[0])

def k_fold_cv(training_images,training_labels,n_folds):
    fold_size = int(np.shape(training_images)[0]/n_folds)
    for num_neighbors in range(1,10,2):

        average_accuracy = 0.0
        for fold in range(n_folds):
            start_ind = int(fold*fold_size)
            end_ind = int((fold+1)*fold_size)

            fold_testing_images = training_images[start_ind:end_ind,:]
            fold_testing_labels = training_labels[start_ind:end_ind, :]

            fold_training_images = np.delete(training_images, np.arange(start_ind,end_ind),0)
            fold_training_labels = np.delete(training_labels, np.arange(start_ind,end_ind),0)

            accuracy = classify_brute_force(fold_training_images, fold_training_labels, fold_testing_images, fold_testing_labels, num_neighbors)

            average_accuracy = average_accuracy + accuracy

        average_accuracy = average_accuracy/n_folds
        print('For ', num_neighbors,' neighbors, the average accuracy is: ',average_accuracy)
8
if __name__ == "__main__":
    training_images, training_labels, testing_images, testing_labels = load_data()
    #k_fold_cv(training_images,training_labels,5)
    classify_brute_force_test(training_images,training_labels,testing_images,testing_labels,1,disp=True)