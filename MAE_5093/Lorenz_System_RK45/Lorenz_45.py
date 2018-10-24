#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
import numpy as np
import matplotlib.pyplot as plt

#Lorenz System of Equations
def rhs(t,u):#RHS function for the Lorenz Equations
    global n_eq
    f = np.zeros(n_eq,dtype='double')
    r = 28.0
    s = 10.0
    b = 8.0/3.0

    f[0] = s*(u[1]-u[0])
    f[1] = r*(u[0]) - u[1] - u[0]*u[2]
    f[2] = u[0]*u[1]-b*u[2]

    return f

#Adaptive time stepping
#Dormand-Prince method (ode45) Method for solving du/dt=rhs
def dp45(a,b,hmin,hmax,sf,tol,u):
    global n_eq

    #ODE45 matrix definitions
    adp45 = np.zeros((7,7),dtype='double')
    cdp45 = np.zeros(7, dtype='double')
    bdp4 = np.zeros(7, dtype='double')
    bdp5 = np.zeros(7, dtype='double')

    adp45[1,0]=1.0/5.0
    adp45[2,0]=3.0/40.0
    adp45[2,1]=9.0/40.0
    adp45[3,0]=44.0/45.0
    adp45[3,1]=-56.0/15.0
    adp45[3,2]=32.0/9.0
    adp45[4,0]=19372.0/6561.0
    adp45[4,1]=-25360.0/2187.0
    adp45[4,2]=64448.0/6561.0
    adp45[4,3]=-212.0/729.0
    adp45[5,0]=9017.0/3168.0
    adp45[5,1]=-355.0/33.0
    adp45[5,2]=46732.0/5247.0
    adp45[5,3]=49.0/176.0
    adp45[5,4]=-5103.0/18656.0
    adp45[6,0]=35.0/384.0
    adp45[6,1]=0.0
    adp45[6,2]=500.0/1113.0
    adp45[6,3]=125.0/192.0
    adp45[6,4]=-2187.0/6784.0
    adp45[6,5]=11.0/84.0


    cdp45[1]=1.0/5.0
    cdp45[2]=3.0/10.0
    cdp45[3]=4.0/5.0
    cdp45[4]=8.0/9.0
    cdp45[5]=1.0
    cdp45[6]=1.0

    bdp4[0]=5179.0/57600.0
    bdp4[1]=0.0
    bdp4[2]=7571.0/16695.0
    bdp4[3]=393.0/640.0
    bdp4[4]=-92097.0/339200.0
    bdp4[5]=187.0/2100.0
    bdp4[6]=1.0/40.0

    bdp5[0]=35.0/384.0
    bdp5[1]=0.0
    bdp5[2]=500.0/1113.0
    bdp5[3]=125.0/192.0
    bdp5[4]=-2187.0/6784.0
    bdp5[5]=11.0/84.0
    bdp5[6]=0.0

    uu = np.zeros(n_eq, dtype='double')#Defining zeros in numpy arrays
    u4 = np.zeros(n_eq, dtype='double')
    u5 = np.zeros(n_eq, dtype='double')
    err = np.zeros(n_eq, dtype='double')


    tiny = 1.0e-8

    dt = hmax
    t = a

    i=0;j=0; #Step counter
    t_rec = np.zeros(1, dtype='double')		#Record arrays
    dt_rec = np.zeros(1, dtype='double')
    u_rec = np.copy([u])

    while t<=(b-tiny):
        # Compute r arrays
        r0 = rhs(t,u)
        uu = u+dt*adp45[1,0]*r0
        tt = t+cdp45[1]*dt

        r1 = rhs(tt,uu)
        uu = u+dt*(adp45[2,0]*r0+adp45[2,1]*r1)
        tt = t+cdp45[2]*dt

        r2 = rhs(tt,uu)
        uu = u + dt * (adp45[3, 0] * r0 + adp45[3, 1] * r1 + adp45[3, 2] * r2)
        tt = t + cdp45[3] * dt

        r3 = rhs(tt,uu)
        uu = u + dt * (adp45[4, 0] * r0 + adp45[4, 1] * r1 + adp45[4, 2] * r2 + adp45[4, 3] * r3)
        tt = t + cdp45[4] * dt

        r4 = rhs(tt, uu)
        uu = u + dt * (adp45[5, 0] * r0 + adp45[5, 1] * r1 + adp45[5, 2] * r2 + adp45[5, 3] * r3 + adp45[5, 4] * r4)
        tt = t + cdp45[5] * dt

        r5 = rhs(tt, uu)
        uu = u + dt * (adp45[6, 0] * r0 + adp45[6, 1] * r1 + adp45[6, 2] * r2 + adp45[6, 3] * r3 + adp45[6, 4] * r4 + adp45[6, 5] * r5)
        tt = t + cdp45[6] * dt

        r6 = rhs(tt, uu)
        #Fourth order solution
        u4 = u + dt * (
        bdp4[0] * r0 + bdp4[1] * r1 + bdp4[2] * r2 + bdp4[3] * r3 + bdp4[4] * r4 + bdp4[5] * r5 + bdp4[6] * r6)
        #Fifth order solution
        u5 = u + dt * (
        bdp5[0] * r0 + bdp5[1] * r1 + bdp5[2] * r2 + bdp5[3] * r3 + bdp5[4] * r4 + bdp5[5] * r5 + bdp5[6] * r6)

        #estimated error per unit time, should be at most tol
        err = np.abs(u5-u4)
        est = np.amax(err)

        if est<=tol or dt<=hmin:
            t = t+dt
            u = u5					#Accept 5th order solution
            i = i+1
            print(t)

        ratio = (sf*tol/(est+tiny))**(1.0/4.0)
        #Step size adjustment
        dt = dt*ratio

        if dt<hmin:
            dt = hmin
        elif dt>hmax:
            dt=hmax

        if t+dt>b:
            dt = b-t

        j=j+1
        print("Number of time steps taken ",i)
        print("Total number of time steps taken ", j)
        print("Number of null steps ", j-i)

        t_rec = np.append(t_rec, [t], axis=0)
        dt_rec = np.append(dt_rec, [dt], axis=0)
        u_rec = np.concatenate((u_rec, [[u[0], u[1], u[2]]]), axis=0)


    del uu,u4,u5
    plt.figure(1)
    plt.title('Lorenz Equations - ODE45')
    plt.xlabel('t')
    plt.ylabel('Function')
    plt.plot(t_rec, u_rec[:, 2], label='u2', color='red')
    plt.plot(t_rec, u_rec[:, 1], label='u1', color='blue')
    plt.plot(t_rec, u_rec[:, 0], label='u0', color='green')
    plt.legend()

    plt.figure(2)
    plt.title('Timestep Values')
    plt.xlabel('t')
    plt.ylabel('dt')
    plt.plot(t_rec, dt_rec, label='dt', color='blue')
    plt.legend()
    plt.show()




if __name__ == "__main__":

    n_eq = 3
    u = np.zeros(n_eq,dtype='double')

    hmin = 1.0e-6
    hmax = 1.0e-2

    tol = 1.0e-7
    sf = 0.5

    a = 0.0
    b = 100.0

    u[0] = 1.0
    u[1] = 0.0
    u[2] = 0.0

    dp45(a,b,hmin,hmax,sf,tol,u)
