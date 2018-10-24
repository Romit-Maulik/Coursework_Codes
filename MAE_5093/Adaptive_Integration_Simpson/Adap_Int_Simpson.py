#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
import numpy as np
import matplotlib.pyplot as plt
import os

#Adaptive Integration using Simpsons base

def function(x):#Function to be integrated
    f = 10.0*np.exp(-50*np.abs(x))-0.01/((x-0.5)*(x-0.5) + 0.001) + 5.0*np.sin(5.0*x)
    return f

#This is the main programs
if __name__ == "__main__":
    exact_integral = -0.56681975015					#Exact answer
    geps = 1.0e-4									#Error criterion
    x_out = np.empty(0,dtype='double')				#Defining empty arrays of size 0
    f_out = np.empty(0,dtype='double')
    f_curve = np.empty(0, dtype='double')
    x_exact = np.empty(0, dtype='double')

    a = -1.0										#Left boundary of domain
    b = 1.0											#Right boundary of domain

    s = 0.0											#Integral summation variable
    x = a
    eps = geps/(b-a)
    h = b - x

    while x<b:
        s1 = h/6.0*(function(x)+4.0*function(x+0.5*h)+function(x+h))
        s2 = h/12.0*(function(x)+4.0*function(x+0.25*h)+2.0*function(x+0.5*h)+4.0*function(x+0.75*h)+function(x+h))

        e1 = 1.0/15.0*np.abs(s2-s1)
        e2 = h*eps

        if e1<=e2:
            x_out = np.append(x_out,x+0.5*h)
            f_out = np.append(f_out,function(x+0.5*h))
            s = s + (16.0*s2-s1)/15.0
            x = x+h
            h = (b-x)
        else:
            h = 0.5*h
	#Exact function plot
    for i in range(0,2001):
        xx = -1.0 + i/2000.0*(b-a)
        x_exact = np.append(x_exact,xx)
        f_curve = np.append(f_curve,function(xx))

    print("Exact: ",exact_integral)
    print("Numerical: ", exact_integral," Error: ",100*np.abs(s-exact_integral)/exact_integral)

    plt.figure()
    plt.title('Adaptive Integration - Simpsons Base')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.plot(x_exact,f_curve,label='Curve')
    plt.scatter(x_out, f_out,color='red',label='Sampling')
    plt.legend()
    plt.show()

