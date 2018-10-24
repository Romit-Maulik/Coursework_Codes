#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
import numpy as np
import matplotlib.pyplot as plt
import os

#Note that there is NO FORCING IN THIS CODE
#Dirichlet TDMA
def tdma(a,b,c,d):

    global nx
    x = np.zeros(nx + 1, dtype='double')

    for i in range(2, nx):              #Will go to nx-1; nx comes from boundary condition; 1 comes from TDMA logic
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    x[nx-1] = d[nx-1]/b[nx-1]

    for i in range(nx-2,0,-1):          #Will go till 1, 0 comes from BC, nx-1 from TDMA logic
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


if __name__ == "__main__":
    #Physics
    alpha=1.0								#Diffusion coefficient
    nx = 20									#Number of points in array	
    dx = 1.0/float(nx)
    dt = 0.001
    time = 0.1									#Final Time
    beta = alpha*dt/(2.0*dx*dx)
    pi = np.pi
    u = np.zeros(nx+1,dtype='double')       ####nx+1 elements starting from zero - 0 to nx
    x = np.linspace(0, 1.0, nx+1,dtype='double')

    #Initial condition
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Crank-Nicolson Method for heat equation')
    u = np.sin(pi*x)
    plt.plot(x,u,label='t=0')

    #TDMA vectors
    a = np.zeros(nx + 1, dtype='double')  ####nx+1 elements starting from zero - 0 to nx
    b = np.zeros(nx + 1, dtype='double')  
    c = np.zeros(nx + 1, dtype='double')  
    d = np.zeros(nx + 1, dtype='double')  



    #Time integration
    t = 0.0
    while t<time:
        t = t+dt

        for i in range(1,nx):
            d[i] = (u[i + 1] - 2.0 * u[i] + u[i - 1]) * beta + u[i]

        a.fill(-beta)
        b.fill(1 + 2.0 * beta)
        c.fill(-beta)

        u = tdma(a,b,c,d)				#Call TDMA for implicit solution

    plt.plot(x, u,label='t=final time')
    plt.legend()
    plt.show()