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


def least_squares_regression(training_data,test_data,poly_order):
    #Distribute basis centers
    basis_centers = np.arange(0.0,60.0,step=60.0/float(poly_order+1))
    sd = 60.0/float(poly_order+1)

    #Make transformed matrix
    phi = np.empty(shape=(np.shape(training_data)[0],poly_order+1))
    for col in range(poly_order+1):
        phi[:, col] = np.exp(-(training_data[:, 0] - basis_centers[col])**2/(2.0*sd**2))

    #Solve for optimal weights
    lhs = np.matmul(np.transpose(phi),phi)
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
    #Load data
    training_data, test_data = load_data()

    #Plot the errors
    fig, ax = plt.subplots(nrows=1,ncols=1)
    ax.set_title('RMS Errors - Least Squares Regression (Radial basis)')
    ax.set_xlabel('Number of basis centers')
    ax.set_ylabel('Mean squared error')

    train_info = []
    test_info = []

    poly_order = 1
    while poly_order <= 20:
        rms_train, rms_test = least_squares_regression(training_data,test_data,poly_order)
        train_info.append([poly_order, rms_train])
        test_info.append([poly_order, rms_test])
        poly_order = poly_order + 1

    train_info = np.asarray(train_info)
    test_info = np.asarray(test_info)

    ax.plot(train_info[:,0],train_info[:,1], color='black',label='Training error',marker='o')
    ax.plot(test_info[:,0],test_info[:,1], color='blue',label='Testing error',marker='o')
    plt.legend()
    plt.show()


def plot_performance(poly_order):

    #Load training data
    training_data, _ = load_data()

    # Distribute basis centers
    basis_centers = np.arange(0.0, 60.0, step=60.0 / float(poly_order + 1))
    sd = 60.0 / float(poly_order + 1)

    # Make transformed matrix
    phi = np.empty(shape=(np.shape(training_data)[0], poly_order + 1))
    for col in range(poly_order + 1):
        phi[:, col] = np.exp(-(training_data[:, 0] - basis_centers[col]) ** 2 / (2.0 * sd ** 2))

    # Solve for optimal weights
    lhs = np.matmul(np.transpose(phi), phi)
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
plot_performance(10)