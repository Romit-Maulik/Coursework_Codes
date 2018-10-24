#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
import numpy as np
import matplotlib.pyplot as plt

# Cubic Spline Interpolation
def spline(xd, fd, x):
    global nd
    a = np.zeros(nd + 1, dtype='double')					#Defining numpy arrays to go from 0 to nd
    b = np.zeros(nd + 1, dtype='double')
    c = np.zeros(nd + 1, dtype='double')
    r = np.zeros(nd + 1, dtype='double')
    g = np.zeros(nd + 1, dtype='double')

    for i in range(1, nd):  # i will go from 1 to 9
        a[i] = (xd[i] - xd[i - 1]) / 6.0
        b[i] = (xd[i + 1] - xd[i - 1]) / 3.0
        c[i] = (xd[i + 1] - xd[i]) / 6.0
        r[i] = (fd[i + 1] - fd[i]) / (xd[i + 1] - xd[i]) - (fd[i] - fd[i - 1]) / (xd[i] - xd[i - 1])

    r[1] = r[1] - a[1] * g[0]
    r[nd - 1] = r[nd - 1] - c[nd - 1] * g[nd]

    z = tdma(a, b, c, r, 1, nd - 1)

    for i in range(1, nd):
        g[i] = z[i]

    for i in range(0, nd):
        if xd[i] <= x <= xd[i + 1]:
            d = xd[i + 1] - xd[i]
            f = g[i] / 6.0*(((xd[i+1] - x) ** 3) / d - d * (xd[i+1] - x)) \
            + g[i+1]/6.0*(((x - xd[i]) ** 3) / d - d * (x - xd[i])) \
            + fd[i] * (xd[i+1] - x) / d + fd[i+1] * (x - xd[i]) / d

    if x < xd[0]:
        i = 0
        d = xd[i + 1] - xd[i]
        f = g[i] / 6.0 * (((xd[i + 1] - x) ** 3) / d - d * (xd[i + 1] - x)) \
        + g[i + 1] / 6.0 * (((x - xd[i]) ** 3) / d - d * (x - xd[i])) \
        + fd[i] * (xd[i + 1] - x) / d + fd[i + 1] * (x - xd[i]) / d
    elif x > xd[nd]:
        i = nd - 1
        d = xd[i + 1] - xd[i]
        f = g[i] / 6.0 * (((xd[i + 1] - x) ** 3) / d - d * (xd[i + 1] - x)) \
        + g[i + 1] / 6.0 * (((x - xd[i]) ** 3) / d - d * (x - xd[i])) \
        + fd[i] * (xd[i + 1] - x) / d + fd[i + 1] * (x - xd[i]) / d

    return f


def tdma(a, b, c, r, s, e):
    global nd
    x = np.zeros(nd + 1, dtype='double')

    for i in range(s + 1, e + 1):
        b[i] = b[i] - a[i] / b[i - 1] * c[i - 1]
        r[i] = r[i] - a[i] / b[i - 1] * r[i - 1]

    x[e] = r[e] / b[e]

    for i in range(e - 1, s - 1, -1):
        x[i] = (r[i] - c[i] * x[i + 1]) / b[i]

    return x


# Main program
if __name__ == "__main__":
    nd = 10  # Number of data points
    n_exact = 2000 #Number of data points for exact solution
    xd = np.zeros(nd + 1, dtype='double')  # xd[0] to xd[nd+1]
    fd = np.zeros(nd + 1, dtype='double')  # fd[0] to fd[nd+1]
    x_exact = np.zeros(n_exact + 1, dtype='double')
    u_exact = np.zeros(n_exact + 1, dtype='double')
    u_spline = np.zeros(n_exact + 1, dtype='double')

    #Clustering data points if needed
    clustering = 0
    if clustering == 1:
        for i in range(0, nd + 1):
            xd[i] = -np.cos(np.pi * i/ (nd))
        for i in range(0, n_exact + 1):
            x_exact[i] = -np.cos(np.pi * i/ (n_exact))
    else:
        for i in range(0, nd + 1):
            xd[i] = -1.0 + i * 2.0 / (nd)
        for i in range(0, n_exact + 1):
            x_exact[i] = -1.0 + i * 2.0 / (n_exact)

    for i in range(0, nd + 1):  # Allocating measurements initially
        fd[i] = 1.0 / (1.0 + 25.0 * xd[i] * xd[i])

    for i in range(0, n_exact + 1):  # Allocating measurements initially
        u_exact[i] = 1.0 / (1.0 + 25.0 * x_exact[i] * x_exact[i])

    x_test = 0.7
    f_exact = 1.0 / (1.0 + 25.0 * x_test * x_test)
    f_test = spline(xd, fd, x_test)

    print('The exact value: ', f_exact)
    print('The cubic spline value: ', f_test)

    #Spline solution for
    for i in range(0,n_exact+1):
        u_spline[i] = spline(xd,fd,x_exact[i])

    #Plotting exact and spline solution
    plt.figure(1)
    plt.title('Cubic Spline Performance')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.plot(x_exact,u_exact,label='Exact')
    plt.plot(x_exact,u_spline,label='Spline')
    plt.scatter(xd, fd, label='Sample',color='red')
    plt.legend()
    plt.show()
