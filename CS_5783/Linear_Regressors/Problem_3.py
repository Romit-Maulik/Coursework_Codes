import numpy as np
import matplotlib.pyplot as plt

#Load data
def load_data():

    data = np.loadtxt('crash.txt')

    #Segregate
    idx = np.arange(0,np.shape(data)[0],2)
    training_data = data[idx,:]

    idx = np.arange(1,np.shape(data)[0],2)
    test_data = data[idx,:]
    return training_data, test_data

def least_squares_regression(training_data,test_data,alpha):
    poly_order = 50

    #Distribute basis centers
    basis_centers = np.arange(0.0,60.0,step=60.0/float(poly_order+1))
    sd = 60.0/float(poly_order+1)

    #Add prior information
    beta = 0.0025

    #Make transformed matrix
    phi = np.empty(shape=(np.shape(training_data)[0],poly_order+1))
    for col in range(poly_order+1):
        phi[:, col] = np.exp(-(training_data[:, 0] - basis_centers[col])**2/(2.0*sd**2))

    #Prior matrix
    prior_mat = alpha/beta*np.identity(np.shape(phi)[1])

    #Solve for optimal weights
    lhs = np.matmul(np.transpose(phi),phi) + prior_mat
    rhs = np.matmul(np.transpose(phi),training_data[:,1])

    w_opt = np.linalg.solve(lhs,rhs)

    #Find RMS error on training data
    pred = np.matmul(phi,w_opt)
    rms_train = np.sum((pred-training_data[:,1])**2,axis=0)
    rms_train = rms_train/np.shape(training_data)[0]

    #Find RMS error on test data
    phi = np.empty(shape=(np.shape(test_data)[0],poly_order+1))
    for col in range(poly_order+1):
        phi[:, col] = np.exp(-(test_data[:, 0] - basis_centers[col])**2/(2.0*sd**2))

    pred = np.matmul(phi,w_opt)
    rms_test = np.sum((pred-test_data[:,1])**2,axis=0)
    rms_test = rms_test / np.shape(test_data)[0]

    return rms_train, rms_test

def plot_rms():
    # Load data
    training_data, test_data = load_data()

    # Plot the errors
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title('RMS Errors - Least Squares Regression (Radial basis)')
    ax.set_xlabel('Natural log of alpha')
    ax.set_ylabel('Mean squared error')

    alpha = np.logspace(-8, 0, 100,base=np.e)

    train_info = []
    test_info = []

    iter = 0
    while iter < np.shape(alpha)[0]:
        alpha_val = alpha[iter]
        rms_train, rms_test = least_squares_regression(training_data, test_data, alpha_val)
        train_info.append([np.log(alpha_val), rms_train])
        test_info.append([np.log(alpha_val), rms_test])
        iter = iter + 1

    train_info = np.asarray(train_info)
    test_info = np.asarray(test_info)

    ax.plot(train_info[:, 0], train_info[:, 1], color='black', label='Training error', marker='o')
    ax.plot(test_info[:, 0], test_info[:, 1], color='blue', label='Testing error', marker='o')
    plt.legend()
    plt.show()

def plot_performance(alpha):
    # Load data
    training_data, _ = load_data()

    poly_order = 50

    # Distribute basis centers
    basis_centers = np.arange(0.0, 60.0, step=60.0 / float(poly_order + 1))
    sd = 60.0 / float(poly_order + 1)

    # Add prior information
    beta = 0.0025

    # Make transformed matrix
    phi = np.empty(shape=(np.shape(training_data)[0], poly_order + 1))
    for col in range(poly_order + 1):
        phi[:, col] = np.exp(-(training_data[:, 0] - basis_centers[col]) ** 2 / (2.0 * sd ** 2))

    # Prior matrix
    prior_mat = alpha / beta * np.identity(np.shape(phi)[1])

    # Solve for optimal weights
    lhs = np.matmul(np.transpose(phi), phi) + prior_mat
    rhs = np.matmul(np.transpose(phi), training_data[:, 1])

    w_opt = np.linalg.solve(lhs, rhs)

    # Find prediction on training data
    pred = np.matmul(phi, w_opt)

    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.set_title('Prediction: Visual Assessment')
    ax.set_xlabel('Time (in ms)')
    ax.set_ylabel('Acceleration')

    ax.plot(training_data[:,0],training_data[:,1],label='Training data')
    ax.plot(training_data[:, 0], pred[:], label='Prediction')

    plt.legend()
    plt.show()

plot_rms()
plot_performance(np.exp(-5))

