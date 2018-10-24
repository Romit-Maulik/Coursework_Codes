#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
# Import statements - these are libraries
import numpy as np  # This imports array data structures
import matplotlib.pyplot as plt  # This imports plot tools
from scipy.integrate import odeint
#import os   #Used for "pause" functionality

def initialize(array):
    global nx
    pi = np.pi

    for i in xrange(0,nx):  # Note that python arrays start at ZERO ONLY syntax: range(start,number of times)
        array[i] = np.sin(2.0 * pi * float(i + 1) / nx)



def spectral_rhs(array,t):

    global alpha
    array_hat = np.fft.fft(array)

    #Spectral prep
    pi = np.pi
    im = complex(0, 1)
    kx = np.array([(2 * pi) * i * im for i in range(0, nx / 2) + [0] + range(-nx / 2 + 1, 0)])
    k2 = kx ** 2

    # Need to calculate first derivative du/dx
    du_hat = kx * array_hat
    du = np.fft.ifft(du_hat)
    udu = array*du
    udu_hat = np.fft.fft(udu)

    return np.real(np.fft.ifft(alpha*k2*array_hat - udu_hat))


def spectral_1D(array,time):
    global alpha, nx, num_op
    h = 1.0/nx
    x = [h*i for i in xrange(1,nx+1)]

    t = np.linspace(0, time, num_op)
    sol = odeint(spectral_rhs, array, t)

    plt.figure(1)
    plt.subplot(211)
    for i in range(0,num_op):
        plt.plot(x, sol[i], label='Time '+str(i))

    plt.title('1D - Viscous Burgers Equation, Spectral - Scipy Integrate')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.grid()




#Main body of execution
if __name__ == "__main__":
    nx = 128
    alpha = 0.01  # Heat equation coefficient
    num_op = 10  # Number of outputs
    u = np.zeros(nx, dtype=np.double)  # The array is of type double precision

    initialize(u)
    spectral_1D(u,0.3)

    u_hat = np.absolute((np.fft.fft(u)))

    plt.subplot(212)
    plt.title('Frequency at final time')

    pi = np.pi
    kx = np.array([(2 * pi) * i for i in range(0, nx / 2) + [0] + range(-nx / 2 + 1, 0)])
    plt.scatter(kx,u_hat,label='Final Time Frequency')
    plt.legend(loc='best')
    plt.xlabel('Wavenumber')
    plt.grid()


    plt.show()