#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
import numpy as np
import matplotlib.pyplot as plt

#Solve y''' + y*y' = 0, for y(x), x in [0,10]

def rk4(x,h,y):
    global n_eq
    k1 = np.zeros(n_eq,dtype='double')
    k2 = np.zeros(n_eq, dtype='double')
    k3 = np.zeros(n_eq, dtype='double')
    k4 = np.zeros(n_eq, dtype='double')

    r = rhs(y,x)

    for i in range(0,n_eq):
        k1[i] = h*r[i]

    r = rhs(y+k1/2.0,x+h/2.0)

    for i in range(0,n_eq):
        k2[i] = h*r[i]

    r = rhs(y+k2/2.0,x+h/2.0)

    for i in range(0,n_eq):
        k3[i] = h*r[i]

    r = rhs(y + k3, x + h)

    for i in range(0, n_eq):
        k4[i] = h * r[i]

    for i in range(0, n_eq):
        y[i] = y[i] + (k1[i]+2*(k2[i]+k3[i])+k4[i])/6.0

    del k1,k2,k3,k4,r


def rhs(y,x):
    global n_eq

    val = np.zeros(n_eq, dtype='double')
    val[0] = -y[0]*y[2]
    val[1] = y[0]
    val[2] = y[1]

    return val


if __name__ == "__main__":
    n_eq = 3                        	#Equations numbered from 0-2
    y = np.zeros(n_eq,dtype='double')

    n = 1000
    a = 0.0
    b = 10.0
    h = (b-a)/float(n)

    #Boundary Conditions
    y1a = 0.0
    y2a = 0.0
    y1b = 1.0

    #initial guesses for y(1) at x=0
    a0 = 1.0
    a1 = 0.5

    #First guess:
    #Initial condition
    y[0] = a0
    y[1] = y1a
    y[2] = y2a
	#Time Integration with RK4 scheme to find b0
    for i in range(1,n+1):
        x = h*i
        rk4(x,h,y)

    b0 = y[1]

    #Second Guess
    #Initial condition
    y[0] = a1
    y[1] = y1a
    y[2] = y2a
	
	#Time Integration with RK4 scheme to find b1
    for i in range(1,n+1):
        x = i*h
        rk4(x,h,y)

    b1 = y[1]
    guess = 0.0

    for i in range(1,101):
        guess = a1+(y1b-b1)/((b1-b0)/(a1-a0))       #Secant method
        y[0] = guess
        y[1] = y1a									#BC
        y[2] = y2a									#BC

        for j in range(1,n+1):
            x = j*h
            rk4(x,h,y)

        b0=b1
        b1=y[1]
        a0=a1
        a1=guess

        #print(i,b1)

        if np.abs(b1-y1b)<=1.0e-6:
            break

    #Final Computation
    y[0]=guess
    y[1] = y1a
    y[2] = y2a

    y_rec = np.copy([y])							#Making numpy arrays to record our solution dynamically
    x_rec = np.zeros(1,dtype='double')

    for i in range(1,n+1):
        x = i*h
        rk4(x,h,y)
        x_rec = np.append(x_rec,[x],axis=0)
        y_rec = np.concatenate((y_rec,[[y[0],y[1],y[2]]]),axis=0)


    plt.figure()
    plt.title('Blasius Boundary Layer - Shooting Method')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_rec,y_rec[:,2],label='f2',color='red')
    plt.plot(x_rec, y_rec[:, 1], label='f1',color='blue')
    plt.plot(x_rec, y_rec[:, 0], label='f0',color='green')
    plt.ylim((-0.5,2))
    plt.xlim((0, 6))
    plt.legend()
    plt.show()