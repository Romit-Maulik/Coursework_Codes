import numpy as np
import matplotlib.pyplot as plt
import os
import gzip as gz


def load_data():

    np.random.seed(10)

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

    #Choose 1000 samples of 5s
    idx = np.ndarray.flatten(np.asarray((np.where(training_labels[:, 0] == 5))))
    idx = np.random.choice(idx, 1000, replace=False)
    five_images = training_images[idx,:]
    five_labels = training_labels[idx,:]
    five_means = np.mean(five_images,axis=0)
    five_vars = np.var(five_images)

    #Choose 1000 samples of not 5s
    idx = np.ndarray.flatten(np.asarray(np.where(training_labels[:, 0] != 5)))
    idx = np.random.choice(idx, 1000, replace=False)
    not_five_images = training_images[idx,:]
    not_five_labels = training_labels[idx, :]
    not_five_means = np.mean(not_five_images,axis=0)
    not_five_vars = np.var(not_five_images)

    images = np.concatenate((five_images,not_five_images),axis=0)
    labels = np.concatenate((five_labels, not_five_labels), axis=0)

    idx = np.random.choice(np.arange(0,2000,1),2000,replace=False)

    testing_images = images[idx, :][1800:, :]
    testing_labels = labels[idx, :][1800:, :]

    # Visualize
    # np.set_printoptions(threshold=np.nan)
    # print(training_images[0,:])
    #
    # check_row = np.reshape(training_images[10,:],(28,28))
    # plt.figure()
    # plt.imshow(check_row)
    # plt.show()
    # print(training_labels[10,:])

    return testing_images, testing_labels, [five_means, not_five_means, five_vars, not_five_vars]

def gaussian_bayes_classifier(testing_images,testing_labels,parameters,tau):
    '''
        Note that there are 784 features in an input vector
        Each feature will have a mean given by a Gaussian conditional probability when 5
        Each feature will have a different mean given by a different Gaussian conditional probability when not 5
        All features have same variance when class is 5
        All features have same variance when class is not 5
    '''
    five_means = parameters[0]
    not_five_means = parameters[1]
    five_vars = parameters[2]
    not_five_vars = parameters[3]

    # Classifier
    label_preds = np.zeros(shape=(np.shape(testing_images)[0], 1), dtype='int')
    correct = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    pos = 0
    neg = 0

    for k in range(np.shape(testing_images)[0]):
        class_probs = np.zeros(shape=(2), dtype='double')

        class_probs[1] = np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * five_vars) * np.exp(
                -((testing_images[k, :] - five_means[:]) ** 2) / (2.0 * five_vars))))

        class_probs[0] = np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * not_five_vars) * np.exp(
            -((testing_images[k, :] - not_five_means[:]) ** 2) / (2.0 * not_five_vars))))

        dec_prob = class_probs[1]-class_probs[0]

        if dec_prob >= tau:
            label_preds[k, 0] = 1
        else:
            label_preds[k, 0] = 0

        if testing_labels[k,0] == 5 and label_preds[k,0] == 1:
            correct = correct + 1
            tp = tp + 1
            pos = pos + 1
        elif testing_labels[k,0] != 5 and label_preds[k,0] == 0:
            correct = correct + 1
            tn = tn + 1
            neg = neg + 1
        elif testing_labels[k,0] != 5 and label_preds[k,0] == 1:
            fp = fp + 1
            neg = neg + 1
        elif testing_labels[k,0] == 5 and label_preds[k,0] == 0:
            fn = fn + 1
            pos = pos + 1

    #print('Accuracy:', correct / 200)
    #print('False positive rate:', fp / pos)
    #print('True positive rate:', tp / pos)

    return fp/pos, tp/pos


if __name__ == "__main__":
    testing_images, testing_labels, parameters = load_data()

    #Plotting ROC curve
    fig, ax = plt.subplots(nrows=1,ncols=1)
    ax.set_title('ROC Curve')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    roc_vals = np.empty(shape=(0,2),dtype='double')
    #gaussian_bayes_classifier(testing_images, testing_labels, parameters, np.log(1 / 1))
    #exit()

    print('Type 1 errors five times as costly: FPR,TPR = ',gaussian_bayes_classifier(testing_images,testing_labels,parameters,np.log(5/1)))
    print('Type 1 errors two times as costly: FPR,TPR = ',gaussian_bayes_classifier(testing_images,testing_labels,parameters,np.log(2/1)))
    print('Type 1 errors equally as costly: FPR,TPR = ',gaussian_bayes_classifier(testing_images,testing_labels,parameters,np.log(1/1)))
    print('Type 2 errors two times as costly: FPR,TPR = ',gaussian_bayes_classifier(testing_images,testing_labels,parameters,np.log(1/2)))
    print('Type 2 errors five times as costly: FPR,TPR = ',gaussian_bayes_classifier(testing_images,testing_labels,parameters,np.log(1/5)))

    for tau in range(200,-200,-2):
        fp, tp = gaussian_bayes_classifier(testing_images,testing_labels,parameters,tau)
        roc_vals = np.append(roc_vals,np.array([[fp, tp]]),axis=0)

    sline = np.zeros(shape=(2,2),dtype='double')
    sline[0,0] = 0.0
    sline[0,1] = 0.0
    sline[1,0] = 1.0
    sline[1,1] = 1.0

    ax.plot(sline[:,0],sline[:,1],linestyle='dashed')
    ax.plot(roc_vals[:,0],roc_vals[:,1],color='red')
    plt.show()


