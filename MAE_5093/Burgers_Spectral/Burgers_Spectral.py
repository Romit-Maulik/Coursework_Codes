#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
# Import statements - these are libraries
import numpy as np  # This imports array data structures
import matplotlib.pyplot as plt  # This imports plot tools
#import os   #Used for "pause" functionality

def initialize(array):
    global nx
    pi = np.pi

    for i in xrange(0,nx):  # Note that python arrays start at ZERO ONLY syntax: range(start,number of times)
        array[i] = np.sin(2.0 * pi * float(i + 1) / nx)

    #Prepare timeplots
    plt.figure(1)
    plt.title('1D - Viscous Burgers Equation, Spectral')
    plt.interactive(False)
    plt.xlabel('x')
    plt.ylabel('Velocity')

    h = 1.0/nx
    x = [h*i for i in xrange(1,nx+1)]
    time_op = 'time = 0'
    plt.plot(x,array,label=time_op)
    #plt.show()
    del x,pi


def spectral_1D(array,time):
    t = 0.0
    global alpha, nx, num_op
    h = 1.0/nx

    #Beginning spectral stuff
    pi = np.pi
    im = complex(0,1)
    kx = np.array([(2*pi)*i*im for i in range(0,nx/2)+[0]+range(-nx/2+1,0)])
    k2 = kx**2

    array_hat = np.fft.fft(array)

    file_op = 1
    while t<time:

        array = np.fft.ifft(array_hat)
        #dt = cfl * h / np.amax(np.absolute(array), axis=0)
        dt = 1.0e-4

        if t+dt > time:
            dt = time - t

        #Need to calculate first derivative du/dx
        du_hat = kx*array_hat
        du = np.fft.ifft(du_hat)
        udu = array*du
        udu_hat = np.fft.fft(udu)

        # We are ready for time integration  - Euler forward for simplicity
        array_hat = array_hat + dt * alpha * k2 * array_hat - dt * udu_hat

        if t > float(file_op) / num_op * time:
            h = 1.0 / nx
            x = [h * i for i in xrange(1, nx + 1)]
            time_op = 'time = ' + str(round(t, 2))
            array = np.real(np.fft.ifft(array_hat))  # back to real space
            plt.plot(x, array, label=time_op)
            file_op = file_op + 1
            del x

        t = t+dt

    x = [h * i for i in xrange(1, nx + 1)]
    time_op = 'time = ' + str(round(t, 2))
    array = np.real(np.fft.ifft(array_hat))  # back to real space
    plt.plot(x, array, label=time_op)

    plt.legend()
    plt.show()
    del x, array_hat, array, kx, k2

#Main body of execution
if __name__ == "__main__":
    nx = 128
    alpha = 0.01  # Heat equation coefficient
    num_op = 10  # Number of outputs
    u = np.zeros(nx, dtype=np.double)  # The array is of type double precision

    initialize(u)
    spectral_1D(u,0.3)