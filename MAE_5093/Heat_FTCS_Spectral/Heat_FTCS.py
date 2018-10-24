#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
# Import statements - these are libraries
import numpy as np  # This imports array data structures
import matplotlib.pyplot as plt  # This imports plot tools
import time as cput  # For timing code


# Initial conditions
def initialize(array):
    length = array.shape[0]  # Number of elements in numpy array
    for i in range(0, length):  # Note that python arrays start at ZERO ONLY syntax: range(start,number of times)
        pi = np.pi
        array[i] = np.sin(2.0 * pi * float(i) / (length - 1))

    # Prepare timeplots
    plt.figure(1)
    plt.interactive(False)
    plt.xlabel('x')
    plt.ylabel('Temperature')
    x = np.linspace(0, 2.0 * np.pi, nx)
    time_op = 'time = 0'
    plt.plot(x, array, label=time_op)
    del x


# Forward time - Central Space time stepping
def ftcs(array, time):
    plt.title('1D - Heat Equation, FTCS')
    t = 0.0  # Initializing time
    global alpha, cfl, nx, num_op, start_time
    dx = 1.0 / float(nx)

    temp_array = np.copy(array)  # New temporary array
    dt = cfl / alpha * dx * dx  # Stability based timestep

    # Time integration
    file_op = 1
    while t < time:

        if t + dt > time:  # Final time step
            dt = time - t

        for i in range(1, nx - 1):  # FTCS Update
            array[i] = temp_array[i] + alpha * dt / (dx * dx) * (
            temp_array[i + 1] + temp_array[i - 1] - 2.0 * temp_array[i])

        t = t + dt
        temp_array = np.copy(array)

        if t > float(file_op) / num_op * time:
            x = np.linspace(0, 2.0 * np.pi, nx)
            time_op = 'time = ' + str(round(t, 2))
            plt.plot(x, array, label=time_op)
            file_op = file_op + 1
            del x

    del temp_array  # Clearing temporary array memory

    print("--- %s seconds ---" % (cput.time() - start_time))

    time_op = 'time = ' + str(round(t, 2))
    x = np.linspace(0, 2.0 * np.pi, nx)
    plt.plot(x, array, label=time_op)
    plt.legend()
    plt.show()  # Show time trends
    del x, array


if __name__ == "__main__":
    nx = 129  # Resolution - numpy array will go from 0 to (nx-1)
    alpha = 0.1  # Heat equation coefficient
    cfl = 0.1  # Stability coefficient
    num_op = 10  # Number of outputs
    temp = np.zeros(nx, dtype=np.double)  # The array is of type double precision

    initialize(temp)  # Initial conditions are added

    start_time = cput.time()
    ftcs(temp, 1.0)  # FTCS solver