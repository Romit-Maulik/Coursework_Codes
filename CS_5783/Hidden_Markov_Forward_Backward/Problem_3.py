import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

def create_forward_process():
    # Track state variable
    # This process starts in the FAIR state
    state = 0 #0-Fair, 1-Loaded

    #Choices for x
    x_range = np.arange(start=1,stop=7,dtype='int')
    #Choices for z
    z_range = np.asarray([0, 1],dtype='int')

    prob_z_f = 1.0/6.0*np.ones(6,dtype='double')
    prob_z_l = 0.1*np.ones(6,dtype='double')
    prob_z_l[5] = 0.5

    #Transition matrix
    z_trans = np.zeros(shape=(2,2),dtype='double')
    z_trans[0, 0] = 0.95  # State change from Fair to Fair (i.e. no change)
    z_trans[0, 1] = 0.05  # State change from Loaded to Fair
    z_trans[1, 0] = 0.10  # State change from Fair to Loaded
    z_trans[1, 1] = 0.90  # State change from Loaded to Loaded

    #Time series outputs
    num_steps = 1000
    process_outputs = np.zeros(shape=num_steps,dtype='int')
    process_states = np.zeros(shape=num_steps, dtype='int')
    for i in range(num_steps):

        process_states[i] = state

        #choose from current state
        if state == 0:
            x_chosen = np.random.choice(x_range,1,p=prob_z_f)
        else:
            x_chosen = np.random.choice(x_range, 1, p=prob_z_l)

        #Choose next state
        if state == 0:
            state = np.random.choice(z_range, 1, p=z_trans[:, 0]/np.sum(z_trans[:, 0]))
        else:
            state = np.random.choice(z_range, 1, p=z_trans[:, 1]/np.sum(z_trans[:, 1]))

        process_outputs[i] = x_chosen

    return process_outputs, process_states

def forward_backward_algorithm(process_outputs,process_states):
    '''
    Note only two states are considered here
    :param process_outputs: Observations (dimension of 6)
    :return: Likely hidden state sequence
    '''
    #Transition matrix
    p_z_zp = np.zeros(shape=(2,2),dtype='float128')
    p_z_zp[0, 0] = 0.95  # State change from Fair to Fair (i.e. no change)
    p_z_zp[0, 1] = 0.05  # State change from Loaded to Fair
    p_z_zp[1, 0] = 0.10  # State change from Fair to Loaded
    p_z_zp[1, 1] = 0.90  # State change from Loaded to Loaded

    #Emission matrix - p(x|z)
    p_x_z = np.zeros(shape=(2, 6), dtype='float128')
    #For fair state
    p_x_z[0,:] = 1.0/6.0
    #For loaded state
    p_x_z[1, :] = 0.1
    p_x_z[1, 5] = 0.5

    num_steps = np.shape(process_outputs)[0]

    #Bishop - forward backward algorithm
    alpha = np.zeros(shape=(2,num_steps),dtype='float128') #Rows are states - 0 is fair
    beta = np.zeros(shape=(2, num_steps),dtype='float128')

    #Forward propagation - time = 1
    alpha[0, 0] = 0.5 * p_x_z[0, process_outputs[0]]
    alpha[1, 0] = 0.5 * p_x_z[1, process_outputs[0]]

    # Recursive forward propagation - Bishop
    for t in range(1,num_steps):
        current_x = process_outputs[t] - 1

        alpha[0, t] = p_x_z[0, current_x] * (
                    alpha[0, t - 1] * (p_z_zp[0, 0]) + alpha[1, t - 1] * (p_z_zp[0, 1]))

        alpha[1, t] = p_x_z[1, current_x] * (
                    alpha[0, t - 1] * (p_z_zp[1, 0]) + alpha[1, t - 1] * (p_z_zp[1, 1]))


    # Recursive backward propagation - Bishop 13.39
    beta[0, num_steps - 1] = 1.0
    beta[1, num_steps - 1] = 1.0

    for t in range(num_steps-2,-1,-1):#Goes to zero

        future_x = process_outputs[t+1] - 1

        beta[0, t] = beta[0, t + 1] * p_x_z[0, future_x] * p_z_zp[0, 0] + beta[1, t + 1] * p_x_z[
            0, future_x] * p_z_zp[1, 0]

        beta[1, t] = beta[0, t + 1] * p_x_z[1, future_x] * p_z_zp[0, 1] + beta[1, t + 1] * p_x_z[
            1, future_x] * p_z_zp[1, 1]

    # Marginal predictions
    marginal_z = np.zeros(shape=(2,num_steps),dtype='float128') #Rows are states - 0 is fair

    # For forward only
    for t in range(0, num_steps):
        marginal_z[0, t] = alpha[0, t] / (alpha[0, t] + alpha[1, t])
        marginal_z[1, t] = alpha[1, t] / (alpha[0, t] + alpha[1, t])

    plt.figure()
    plt.plot(marginal_z[1,:],label='Forward estimate')
    plt.plot(process_states[:], label='True states')
    plt.legend()
    plt.show()

    marginal_z = np.zeros(shape=(2, num_steps), dtype='double')  # Rows are states

    # For both forward and backward
    for t in range(0, num_steps):
        marginal_z[0, t] = alpha[0, t] * beta[0,t] / (alpha[0, t]*beta[0,t] + alpha[1, t]*beta[1,t])
        marginal_z[1, t] = alpha[1, t] * beta[1,t] / (alpha[0, t]*beta[0,t] + alpha[1, t]*beta[1,t])

    plt.figure()
    plt.plot(marginal_z[1, :], label='Forward (and backward) estimate')
    plt.plot(process_states[:], label='True states')
    plt.legend()
    plt.show()


process_outputs, process_states = create_forward_process()
forward_backward_algorithm(process_outputs,process_states)




