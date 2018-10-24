#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
import numpy as np
import matplotlib.pyplot as plt
import os

#Assuming Periodic Boundary Conditions - will use FFT - boundary domains are 1.0
def initialize(array,exact):
    global nx,pi
    h = 1.0/nx
    x = [h * i for i in xrange(1, nx + 1)]
    y = [h * i for i in xrange(1, nx + 1)]

    for i in range(0,nx):
        for j in range(0,nx):
            exact[i,j] = np.cos(2.0*pi*x[i]) + np.cos(2.0*pi*y[j])
            array[i,j] = -4.0*pi*pi*(np.cos(2.0*pi*x[i]) + np.cos(2.0*pi*y[j]))



def spectral_poisson_2D(array):
    global nx,pi
    #Setting up wavenumbers
    im = complex(0,1)
    kx = np.array([(2*pi)*y*im for y in range(0,nx/2)+[0]+range(-nx/2+1,0)])
    ky = kx

    kxx = np.zeros((nx, nx), dtype=complex)
    kyy = np.zeros((nx, nx), dtype=complex)

    for i in range(0,nx):
        for j in range(0,nx):
            kyy[i, j] = ky[j] ** 2
            kxx[i, j] = kx[i] ** 2

    #Taking fft2d for source term
    array_hat = np.fft.fft2(array)
    array_hat = array_hat/(kxx+kyy+10**(-10))

    array_hat[0,0]=0.0
    array_hat[nx/2,nx/2] = 0.0
    array_hat[nx/2,0] = 0.0
    array_hat[0, nx/2] = 0.0

    sol = np.real(np.fft.ifft2(array_hat))
    return sol


if __name__ == "__main__":
    nx = 64 #Resolution array from 0 to nx-1, nxth point is 0
    pi = np.pi
    rhs = np.arange((nx)*(nx),dtype='float').reshape(nx,nx)
    exact = np.arange((nx) * (nx),dtype='float').reshape(nx, nx)

    initialize(rhs,exact)
    solution = spectral_poisson_2D(rhs)


    #Plotting
    h = 1.0 / nx
    x = [h * i for i in xrange(1, nx + 1)]
    y = [h * i for i in xrange(1, nx + 1)]
    levels = np.linspace(-2, 2, 20)

    plt.figure(1)
    solplot = plt.contourf(x,y,solution,levels=levels)
    plt.colorbar(solplot, format="%.2f")
    plt.title("Numerical Solution")
    plt.show()

    exact_plot = plt.contourf(x,y,exact,levels=levels)
    plt.colorbar(exact_plot, format="%.2f")
    plt.title("Exact Solution")
    plt.show()

    norm_plot = plt.contourf(x, y, np.absolute(exact-solution))
    plt.colorbar(norm_plot)
    plt.title("L1 Norm")
    plt.show()


    #print(exact[0,0])
    #print(exact[nx-1,nx-1])
    #print(solution[0,0])
    #print(solution[nx-1,nx-1])
