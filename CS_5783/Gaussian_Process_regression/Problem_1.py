import numpy as np
import matplotlib.pyplot as plt
import time

def load_data():

    data = np.loadtxt('crash.txt')

    #Scale data to between -1 and 1
    max_range = np.ndarray.max(data, axis=0)
    norm_data = (data) / (max_range)  # Between -1 and 1

    return norm_data, max_range

def squared_exponential_kernel(i,j,data,sigma):
    return np.exp(-((data[i,0]-data[j,0])**2)/(2*sigma*sigma))

def exponential_kernel(i,j,data,sigma):
    return np.exp(-(np.abs(data[i,0]-data[j,0]))/(sigma))

def squared_exponential_kernel_predict(indices,data,sigma,new_data_point):
    vec1 = np.exp(-((data[indices,0]-new_data_point)**2)/(2*sigma*sigma))
    return vec1

def exponential_kernel_predict(indices,data,sigma,new_data_point):
    vec1 = np.exp(-(np.abs(data[indices,0]-new_data_point))/(sigma))
    return vec1

def construct_c_matrix(data,sigma):
    num_data_points = np.shape(data)[0]

    #Generating meshgrid for vectorized computation
    i_indices, j_indices = np.meshgrid(np.arange(num_data_points,dtype='int'), np.arange(num_data_points,dtype='int'))

    # Calculating gram-matrix
    if kernel_flag == 'Exponential':
        gram_matrix = exponential_kernel(i_indices,j_indices,data,sigma)
    else:
        gram_matrix = squared_exponential_kernel(i_indices, j_indices, data, sigma)

    #Adding diagonal noise
    beta = 1.0 / ((20.0/75) ** 2)
    noise_vector = beta*np.ones(shape=(num_data_points),dtype='double')

    c_matrix = gram_matrix + np.diag(noise_vector)

    return c_matrix

def gpr_predict(data,c_matrix,new_data_point,sigma):
    num_data_points = np.shape(data)[0]
    indices = np.arange(num_data_points,dtype='int')

    if kernel_flag == 'Exponential':
        kvec = exponential_kernel_predict(indices,data,sigma,new_data_point)
    else:
        kvec = squared_exponential_kernel_predict(indices, data, sigma, new_data_point)

    kvec = np.reshape(kvec,newshape=(np.shape(kvec)[0],1))

    #Make prediction
    cmat_inv =  np.linalg.inv(c_matrix)
    predicted_mean = np.matmul(np.matmul(np.transpose(kvec),cmat_inv),data[:,1])

    return predicted_mean

def five_fold_cv(data,n_folds):
    #Separate data into folds
    num_samples = np.shape(data)[0]
    fold_size = num_samples/n_folds

    metric = 0.0

    for fold in range(n_folds):
        start_ind = int(fold * fold_size)
        end_ind = int((fold + 1) * fold_size)

        fold_test_data = data[start_ind:end_ind,:]
        fold_train_data = np.concatenate((data[0:start_ind,:],data[end_ind:,:]),axis=0)

        c_matrix = construct_c_matrix(fold_train_data,sigma)
        new_preds = np.zeros(shape=(np.shape(fold_test_data)[0],2),dtype='double')
        new_preds[:,0] = fold_test_data[:,0]

        #Predict for test fold
        for point in range(np.shape(fold_test_data)[0]):
            new_point = new_preds[point, 0]
            new_preds[point, 1] = gpr_predict(fold_train_data, c_matrix, new_point,sigma)

        #Return a score for accuracy (MSE)
        metric = metric + np.sum((new_preds[:,1]-fold_test_data[:,1])**2,axis=0)

        del new_preds, fold_test_data, fold_train_data

    return metric

#Main function
#Global variables
kernel_flag = 'Sqaured-exponential' #'Exponential' or 'Squared-exponential'
sigma = 0.101 #
data, max_range = load_data()
c_matrix = construct_c_matrix(data,sigma)

num_new_points = 2000
point_diff = 1.0/num_new_points
new_preds = np.zeros(shape=(num_new_points,2),dtype='double')
new_preds[:,0] = point_diff*np.arange(0,num_new_points,dtype='double')

for point in range(num_new_points):
    new_point = new_preds[point,0]
    new_preds[point,1] = gpr_predict(data,c_matrix,new_point,sigma)

#Plot the prediction (these are unscaled)
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.scatter(max_range[0]*data[:,0],max_range[1]*data[:,1],label='Data',s=10)
ax.plot(max_range[0]*new_preds[:,0],max_range[1]*new_preds[:,1],label='Prediction')
plt.legend()
plt.show()
exit()

#5-fold cross validation for a range of sigmas
sigma_vals = np.arange(0.001,1.0,step=1.0/1000)
best_metric = 1.0e10
optimal_sigma = 0.0

for i in range(np.shape(sigma_vals)[0]):
    sigma = sigma_vals[i]
    metric = five_fold_cv(data,5)

    if (metric<best_metric):
        best_metric = metric
        optimal_sigma = sigma

    print(best_metric,optimal_sigma)

