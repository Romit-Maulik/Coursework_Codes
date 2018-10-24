#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
# Import statements - these are libraries
import numpy as np  # This imports array data structures
import matplotlib.pyplot as plt  # This imports plot tools
import time as cput  # For timing code


# Initial conditions
def initialize(array):

    global nx

    for i in range(0, nx):  # Note that python arrays start at ZERO ONLY syntax: range(start,number of times)
        pi = np.pi
        array[i] = np.sin(2.0 * pi * float(i+1) / nx)

    # Prepare timeplots
    plt.figure(1)
    plt.interactive(False)
    plt.xlabel('x')
    plt.ylabel('Temperature')

    h = 1.0 / nx
    x = [h * i for i in xrange(1, nx + 1)]

    time_op = 'time = 0'
    plt.plot(x, array, label=time_op)
    del x


# Spectral method using Cooley-Tukey FFT
def spectral_1D(array, time):
    plt.title('1D - Heat Equation, Spectral')
    t = 0.0  # Initializing time
    global alpha, cfl, nx, num_op, start_time  # Global Variables
    dx = 1.0 / float(nx)  # Spatial discretization - Only for timestep
    dt = cfl / alpha * dx * dx  # Stability based timestep

    im = complex(0, 1)
    k = np.array([(2 * np.pi) * im * y for y in range(0, nx/2) + [0] + range(-nx / 2 + 1, 0)])
    k2 = k ** 2
    array_hat = np.fft.fft(array)

    # Time integration
    file_op = 1
    while t < time:

        if t + dt > time:  # Final time step
            dt = time - t

        array_hat = array_hat + dt * alpha * k2 * array_hat

        if t > float(file_op) / num_op * time:
            h = 1.0 / nx
            x = [h * i for i in xrange(1, nx + 1)]
            time_op = 'time = ' + str(round(t, 2))
            array = np.real(np.fft.ifft(array_hat))  # back to real space
            plt.plot(x, array, label=time_op)
            file_op = file_op + 1
            del x

        t = t + dt

    h = 1.0 / nx
    x = [h * i for i in xrange(1, nx + 1)]
    time_op = 'time = ' + str(round(t, 2))
    array = np.real(np.fft.ifft(array_hat))  # back to real space
    plt.plot(x, array, label=time_op)

    print("--- %s seconds ---" % (cput.time() - start_time))

    plt.legend()
    plt.show()
    del x, array_hat, array, k, k2


if __name__ == "__main__":
    nx = 128  # Resolution - numpy array will go from 0 to (nx-1)
    alpha = 0.1  # Heat equation coefficient
    cfl = 0.1  # Stability coefficient
    num_op = 10  # Number of outputs
    temp = np.zeros(nx, dtype=np.double)  # The array is of type double precision

    initialize(temp)  # Initial conditions are added

    start_time = cput.time()
    spectral_1D(temp, 1.0)  # Spectral solver
