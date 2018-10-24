import numpy as np
import matplotlib.pyplot as plt
import os
import gzip as gz

def load_data(train_subset=12000,test_subset = 2000):
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

    #Threshold training and test inputs
    training_images[training_images[:, :] < 128.0] = 0
    testing_images[testing_images[:, :] < 128.0] = 0
    training_images[training_images[:,:]>128.0] = 1
    testing_images[testing_images[:,:]>128.0] = 1


    #Random shuffling
    randomize = np.arange(np.shape(training_images)[0])
    np.random.shuffle(randomize)
    training_images = training_images[randomize]
    training_labels = training_labels[randomize]

    # Visualize
    # np.set_printoptions(threshold=np.nan)
    # print(training_images[0,:])
    #
    # check_row = np.reshape(training_images[0,:],(28,28))
    # plt.figure()
    # plt.imshow(check_row)
    # plt.show()

    return training_images[:train_subset,:], training_labels[:train_subset,:], testing_images[:test_subset,:], testing_labels[:test_subset,:]

def naive_bayes_classifier(training_images,training_labels,testing_images,testing_labels):
    '''
    Note that there are 784 features in an input vector
    Need to calculate probability of each feature being in one class out of 10 for all training data
    '''
    dirichlet_prior = np.zeros(shape=(10),dtype='double')#Dirichlet prior
    for class_val in range(10):
        idx = np.ndarray.flatten(np.asarray((np.where(training_labels[:, 0] == class_val))))
        num_vals = np.shape(idx)[0]
        dirichlet_prior[class_val] = num_vals/np.shape(training_images)[0]

    #Choose a sample test image
    cond_probs = np.zeros(shape=(784,10),dtype='double')#Conditional probability of activated pixel

    #Finding conditional probabilities
    for class_val in range(10):
        idx = np.ndarray.flatten(np.asarray((np.where(training_labels[:, 0] == class_val))))
        cond_probs[:,class_val] = np.count_nonzero(training_images[idx,:],axis=0)
        cond_probs[:,class_val] = cond_probs[:,class_val]/np.shape(training_images)[0]

    #Classifier
    label_preds = np.zeros(shape=(np.shape(testing_images)[0],1),dtype='int')
    correct = 0

    for k in range(np.shape(testing_images)[0]):
        class_probs = np.zeros(shape=(10), dtype='double')
        for class_val in range(10):
            class_probs[class_val] = dirichlet_prior[class_val] + np.sum(np.log(cond_probs[:, class_val][np.where(testing_images[k, :] > 0)]))

        label_preds[k,0] = np.argmax(class_probs)

        if label_preds[k,0] == testing_labels[k,0]:
            correct = correct + 1

    print('Accuracy:',correct/np.shape(testing_images)[0])

#Main function
if __name__ == "__main__":
    training_images, training_labels, testing_images, testing_labels = load_data(60000,10000)
    naive_bayes_classifier(training_images,training_labels,testing_images,testing_labels)
