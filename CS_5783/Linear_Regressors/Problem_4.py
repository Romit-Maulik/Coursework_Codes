import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load_data():

    np.random.seed(10)

    #Load data from text file
    irises = np.loadtxt('iris.data',delimiter=',',dtype='str')

    irises_inputs = np.array(irises[:,0:4])
    irises_inputs = irises_inputs.astype(np.float)

    #Code string to float
    irises_outputs = irises[:,4]
    irises_outputs[np.where(irises_outputs[:] == r'Iris-setosa')] = str(0)
    irises_outputs[np.where(irises_outputs[:] == r'Iris-versicolor')] = str(1)
    irises_outputs[np.where(irises_outputs[:] == r'Iris-virginica')] = str(2)
    irises_outputs = np.reshape(irises_outputs.astype(np.int32),newshape=(np.shape(irises_outputs)[0],1))

    #Add column of ones to the input data
    irises_inputs = np.concatenate((np.ones(shape=(np.shape(irises_inputs)[0],1)),irises_inputs),axis=1)

    #One hot encoding
    irises_labels = np.zeros(shape=(np.shape(irises_outputs)[0],3),dtype='double')
    mask = irises_outputs[:,0]

    for i in range(np.shape(irises_labels)[0]):
        irises_labels[i,mask[i]] = 1.0

    #Segregate into training and test
    idx = np.arange(0, np.shape(irises_inputs)[0], 2)
    training_inputs = irises_inputs[idx, :]
    training_labels = irises_labels[idx, :]

    idx = np.arange(1, np.shape(irises_inputs)[0], 2)
    test_inputs = irises_inputs[idx, :]
    test_labels = irises_labels[idx, :]

    return training_inputs, training_labels, test_inputs, test_labels


def multiclass_logistic_regression():

    global training_labels, training_inputs
    global test_labels, test_inputs

    num_classes = np.shape(training_labels)[1]
    num_features = np.shape(training_inputs)[1]

    weights = np.ones(shape=(num_features*num_classes),dtype='double')

    def softmax_error(weights):
        global training_labels, training_inputs

        # Finding softmax transformation
        a1 = np.reshape(np.sum(weights[0:5] * training_inputs[:, :], axis=1),
                        newshape=(np.shape(training_inputs)[0], 1))
        a2 = np.reshape(np.sum(weights[5:10] * training_inputs[:, :], axis=1),
                        newshape=(np.shape(training_inputs)[0], 1))
        a3 = np.reshape(np.sum(weights[10:15] * training_inputs[:, :], axis=1),
                        newshape=(np.shape(training_inputs)[0], 1))

        amat = np.concatenate((a1, a2, a3), axis=1)
        ymat = np.copy(amat)

        ymat[:, 0] = np.exp(amat[:, 0]) / (np.exp(amat[:, 0]) + np.exp(amat[:, 1]) + np.exp(amat[:, 2]))
        ymat[:, 1] = np.exp(amat[:, 1]) / (np.exp(amat[:, 0]) + np.exp(amat[:, 1]) + np.exp(amat[:, 2]))
        ymat[:, 2] = np.exp(amat[:, 2]) / (np.exp(amat[:, 0]) + np.exp(amat[:, 1]) + np.exp(amat[:, 2]))

        #Prior for stabilization
        alpha = np.exp(-5)
        prior_val = alpha*np.sum(weights**2,axis=0)

        #Finding error function - Equation 4.108 - Bishop
        softmax_error_val = prior_val-np.sum(training_labels[:, 0] * np.log(ymat[:, 0]) + training_labels[:, 1] * np.log(
            ymat[:, 1]) + training_labels[:, 2] * np.log(ymat[:, 2]),axis=0)
        return softmax_error_val

    w_hat = minimize(softmax_error,weights,options={'disp':True}).x


    #Prediction on testing data
    # Finding softmax transformation
    z1 = np.reshape(np.sum(w_hat[0:5] * training_inputs[:, :], axis=1),
                    newshape=(np.shape(training_inputs)[0], 1))
    z2 = np.reshape(np.sum(w_hat[5:10] * training_inputs[:, :], axis=1),
                    newshape=(np.shape(training_inputs)[0], 1))
    z3 = np.reshape(np.sum(w_hat[10:15] * training_inputs[:, :], axis=1),
                    newshape=(np.shape(training_inputs)[0], 1))

    zmat = np.concatenate((z1, z2, z3), axis=1)
    smat = np.copy(zmat)

    smat[:, 0] = np.exp(zmat[:, 0]) / (np.exp(zmat[:, 0]) + np.exp(zmat[:, 1]) + np.exp(zmat[:, 2]))
    smat[:, 1] = np.exp(zmat[:, 1]) / (np.exp(zmat[:, 0]) + np.exp(zmat[:, 1]) + np.exp(zmat[:, 2]))
    smat[:, 2] = np.exp(zmat[:, 2]) / (np.exp(zmat[:, 0]) + np.exp(zmat[:, 1]) + np.exp(zmat[:, 2]))

    classification_pred = np.argmax(smat,axis=1)
    classification_true = np.argmax(test_labels, axis=1)

    correct = 0
    for i in range(np.shape(classification_true)[0]):
        if classification_true[i] == classification_pred[i]:
            correct = correct + 1

    print('Accuracy of logistic regression:',100.0*correct/np.shape(classification_pred)[0],'%')



training_inputs, training_labels, test_inputs, test_labels = load_data()
multiclass_logistic_regression()