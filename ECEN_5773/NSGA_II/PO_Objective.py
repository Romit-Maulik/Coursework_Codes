import numpy as np
import matplotlib.pyplot as plt
import os
import random

class Front:
    def __init__(self,i):
        self.number = i
        self.data = np.empty(shape=(0,),dtype='int')#List of indexes which this Front owns

class Player:
    def __init__(self,i):
        self.number = i
        self.SP = np.empty(shape=(0,),dtype='int')#list of indexes which are dominated by player
        self.rank = 0
        self.NP = 0

class temp_member:
    def __init__(self):
        self.Q = np.empty(shape=(0,),dtype='int')#list of indexes which are dominated by player

def scheme_objective_function(ap3,am3):
    global nx, kxval, tol, max_dissipation, max_dispersion_deviation

    ap3 = -ap3
    am3 = -am3

    a0 = -10.0 * (am3 + ap3)
    ap1 = 5.0 * am3 + 10.0 * ap3 - 2.0 / 3.0
    ap2 = -am3 - 5.0 * ap3 + 1.0 / 12.0
    am1 = 10.0 * am3 + 5.0 * ap3 + 2.0 / 3.0
    am2 = -5.0 * am3 - ap3 - 1.0 / 12.0

    a0 = -a0
    ap1 = -ap1
    ap2 = -ap2
    ap3 = -ap3
    am1 = -am1
    am2 = -am2
    am3 = -am3

    im = complex(0, 1)
    k_x = np.zeros(nx + 1, dtype='double')

    start_index = 0
    for i in range(0, nx + 1):
        k_x[i] = np.pi * i / float(nx)
        if k_x[i] < kxval:
            start_index = start_index + 1

    mwe_complex = am3 * (np.cos(3.0 * k_x) - im * np.sin(3.0 * k_x)) + am2 * (
        np.cos(2.0 * k_x) - im * np.sin(2.0 * k_x)) + am1 * (np.cos(k_x) - im * np.sin(k_x))

    mwe_complex = mwe_complex + a0 + ap1 * (np.cos(k_x) + im * np.sin(k_x)) + ap2 * (
        np.cos(2.0 * k_x) + im * np.sin(2.0 * k_x)) + ap3 * (
        np.cos(3.0 * k_x) + im * np.sin(3.0 * k_x))

    dispersion = np.zeros((nx + 1), dtype='double')
    dissipation = np.zeros((nx + 1), dtype='double')

    for i in range(0, nx + 1):
        dissipation[i] = np.real(mwe_complex[i])
        dispersion[i] = np.imag(mwe_complex[i])

    disp_error = 0.0
    diss_error = 0.0

    for i in range(0,nx+1):
         if dissipation[i]<-tol:
             diss_error = np.inf
         elif dissipation[i]>max_dissipation:
             diss_error = np.inf

    for i in range(start_index, nx + 1):
        diss_error = diss_error + np.abs(dissipation[i] - 0.0)

    for i in range(0,nx+1):
        if k_x[i]-dispersion[i]<max_dispersion_deviation:
            disp_error = np.inf
            diss_error = np.inf
        elif dispersion[i]<-tol:
            disp_error = np.inf
            diss_error = np.inf

    for i in range(start_index+1, nx + 1):
        disp_error = disp_error + 1.0/k_x[i]*np.abs(k_x[i] - dispersion[i])

    return disp_error, diss_error

def fast_non_dominated_sort(nplayers,of_array):
    playerlist = []
    frontlist = []

    y = Front(0)
    for i in range(nplayers):

        x = Player(i)
        for j in range(nplayers):
            if i != j:
                if of_array[j,0]>of_array[i,0] and of_array[j,1]<of_array[i,1]:#Need to change logic here
                    x.SP = np.append(x.SP,[j,])
                elif of_array[j,0]<of_array[i,0] and of_array[j,1]>of_array[i,1]:
                    x.NP = x.NP + 1

        if x.NP == 0:
            x.rank = 1
            y.data = np.append(y.data,[i,])
        playerlist.append(x)


    frontlist.append(y)#Correct till here

    for i in range(nplayers):
        y = frontlist[i]
        if y.data.size:
            z = temp_member()
            for l in range(np.shape(y.data)[0]):
                pval = y.data
                for count in range(nplayers):
                    if pval[l] == playerlist[count].number:
                        splist = playerlist[count].SP

                        for q in range(np.shape(splist)[0]):
                            playerlist[splist[q]].NP = playerlist[splist[q]].NP - 1
                            if playerlist[splist[q]].NP == 0:
                                playerlist[splist[q]].rank = i+1
                                z.Q = np.append(z.Q,[splist[q],])

            ynext = Front(i+1)
            ynext.data = np.append(ynext.data,z.Q)
            frontlist.append(ynext)
        else:
            ynext = Front(i+1)
            frontlist.append(ynext)

    del playerlist
    return frontlist

def crowding_sort(frontlist,of_array):

    #Dimension 1
    front_vals = np.asarray(frontlist,dtype='int')
    ofvals = np.zeros((np.shape(front_vals)[0],2),dtype='double')
    for i in range(np.shape(front_vals)[0]):
        ofvals[i,0] = np.int(front_vals[i])
        ofvals[i,1] = of_array[np.int(ofvals[i,0]),0]

    ofvals = ofvals[np.argsort(ofvals[:,1])]
    ofvals = np.flip(ofvals,0)#Sorted in descending order

    dist_vals_zero = np.zeros((np.shape(front_vals)[0],2),dtype='double')
    for i in range(0,(np.shape(front_vals)[0])):
        dist_vals_zero[i,0] = ofvals[i,0]

    dist_vals_zero[0,1] = np.inf
    dist_vals_zero[len(frontlist)-1,1] = np.inf

    for i in range(1,(np.shape(front_vals)[0])-1):
        dist_vals_zero[i,1]=dist_vals_zero[i,1]+(ofvals[i+1,1]-ofvals[i-1,1])/(np.amax(ofvals)-np.amin(ofvals))


    #Dimension 2
    for i in range(np.shape(front_vals)[0]):
        ofvals[i, 0] = np.int(front_vals[i])
        ofvals[i, 1] = of_array[np.int(ofvals[i, 0]), 1]

    ofvals = ofvals[np.argsort(ofvals[:, 1])]
    ofvals = np.flip(ofvals,0)  # Sorted in descending order

    dist_vals_one = np.zeros((np.shape(front_vals)[0],2), dtype='double')
    for i in range(0, np.shape(front_vals)[0]):
        dist_vals_one[i, 0] = ofvals[i, 0]

    dist_vals_one[0, 1] = np.inf
    dist_vals_one[np.shape(front_vals)[0] - 1, 1] = np.inf

    for i in range(1, np.shape(front_vals)[0] - 1):
        dist_vals_one[i, 1] = dist_vals_one[i, 1] + (ofvals[i + 1,1] - ofvals[i - 1,1]) / (
        np.amax(ofvals) - np.amin(ofvals))


    return_val = np.zeros((np.shape(front_vals)[0],2), dtype='double')

    #To return front values and distances
    for i in range(np.shape(front_vals)[0]):
        index = np.int(front_vals[i])
        return_val[i,0] = index
        break_val = 0
        for j in range(np.shape(front_vals)[0]):
            if np.int(dist_vals_one[j, 0]) == index:
                return_val[i,1] = return_val[i,1]+ dist_vals_one[j,1]
                break_val = break_val+1
                if break_val == 2:
                    break

            if np.int(dist_vals_zero[j, 0]) == index:
                return_val[i,1] = return_val[i,1]+ dist_vals_zero[j,1]
                break_val = break_val + 1
                if break_val == 2:
                    break

    #Crowding distances are available now
    return return_val

def plot_front(int_val, of_array,front_vals,crowd_vals):

    plt.figure(1)
    plt.scatter(of_array[:, 0], of_array[:, 1], label='Players')
    plt.title('Objective Function Values')
    plt.xlabel('Dispersion Error')
    plt.ylabel('Dissipation Error')

    plt.scatter(of_array[front_vals, 0], of_array[front_vals, 1], label='Front '+str(int_val))
    # for i in range(np.shape(front_vals)[0]):
    #     index = front_vals[i]
    #     abc = str('%s' % float('%.2f' % of_array[index,0]))+str(',')+str('%s' % float('%.2f' % of_array[index,1]))
    #     plt.annotate(abc,(of_array[index, 0], of_array[index, 1]))

    plt.legend()
    plt.show()


    # plt.figure(2)
    # plt.scatter(of_array[:, 0], of_array[:, 1], label='Players')
    # plt.title('Crowding Distances')
    # plt.xlabel('Dispersion Error')
    # plt.ylabel('Dissipation Error')
    #
    # plt.scatter(of_array[front_vals, 0], of_array[front_vals, 1], label='Front '+str(int_val))
    #
    # for i in range(np.shape(front_vals)[0]):
    #     index = np.int(crowd_vals[i,0])
    #     abc = str('%s' % float('%.2f' % crowd_vals[i,1]))
    #     plt.annotate(abc,(of_array[index, 0], of_array[index, 1]))
    #
    # plt.legend()
    # plt.show()

def player_initialize(nplayers,nx):
    player_vals = np.zeros((nplayers, 2), dtype='double')
    of_array = np.zeros((nplayers, 2), dtype='double')

    i = 0
    while i < nplayers:

        a = np.random.uniform(-1.0,1.0)
        b = np.random.uniform(-1.0,1.0)

        val1, val2 = scheme_objective_function(a, b)

        if val1 != np.inf and val2 != np.inf:
            player_vals[i, 0] = a
            player_vals[i, 1] = b

            of_array[i, 0] = val1
            of_array[i, 1] = val2
            i = i + 1

    return of_array, player_vals

def nsga(player_vals,of_array):
    global nplayers
    new_array = np.zeros((np.shape(player_vals)[0]*2,2), dtype='double')
    double_of_array = np.zeros((np.shape(of_array)[0] * 2, 2), dtype='double')

    #Crossover
    global sbx_param_n
    cross_id = 0

    while cross_id !=np.shape(new_array)[0]:
        idx = np.random.randint(np.shape(player_vals)[0],size=2)

        p1 = player_vals[idx[0],:]
        p2 = player_vals[idx[1], :]

        u = np.random.uniform(0.0, 1.0)
        beta = 0.0

        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (sbx_param_n + 1))
        else:
            beta = (1.0 / (2.0 - 2.0 * u)) ** (1.0 / (sbx_param_n + 1))

        c1 = 0.5 * (p1 + p2) - 0.5 * beta * (p2 - p1)
        c2 = 0.5 * (p2 + p2) + 0.5 * beta * (p2 - p1)

        val1c1, val2c1 = scheme_objective_function(c1[0], c1[1])
        val1c2, val2c2 = scheme_objective_function(c2[0], c2[1])

        if val1c1 != np.inf and val2c1 != np.inf and val1c2 != np.inf and val2c2 != np.inf:
            new_array[cross_id,:] = c1[:]
            new_array[cross_id+1,:] = c2[:]

            double_of_array[cross_id, 0] = val1c1
            double_of_array[cross_id, 1] = val2c1

            double_of_array[cross_id+1, 0] = val1c2
            double_of_array[cross_id+1, 1] = val2c2

            cross_id = cross_id + 2

    #Crossover complete
    global mut_rate
    mut_indices = random.sample(range(np.shape(new_array)[0]),
                                int(mut_rate * np.shape(new_array)[0]))  # Generate random numbers which are not duplicate

    for i in range(np.shape(mut_indices)[0]):
        index = mut_indices[i]
        p1 = new_array[index, :]

        for j in range(np.shape(p1)[0]):
            p1[j]=p1[j]*np.random.uniform(-1.0,1.0)

        val1, val2 = scheme_objective_function(p1[0], p1[1])

        if val1 != np.inf and val2 != np.inf:
            new_array[index, :] = p1[:]
            double_of_array[index, 0] = val1
            double_of_array[index, 1] = val2
        else:
            i = i-1

    #Mutation Complete

    #Step 1 organize double array into fronts
    list_of_fronts = fast_non_dominated_sort(2*nplayers, double_of_array)
    size = 0
    frontid = 0
    while size < nplayers:
        size = size + np.shape(list_of_fronts[frontid].data)[0]
        frontid = frontid+1

    carry_over_indices = np.empty(0,dtype='int')

    for i in range(frontid-1):
        carry_over_indices = np.concatenate((carry_over_indices,list_of_fronts[i].data),axis=0)

    rem_indices = np.shape(of_array)[0]-np.shape(carry_over_indices)[0]

    crowd_vals = crowding_sort(list_of_fronts[frontid-1].data,double_of_array)
    crowd_vals = crowd_vals[np.argsort(crowd_vals[:, 1])]
    crowd_vals = np.flip(crowd_vals, 0)  # Sorted in descending order



    for i in range(rem_indices):
        index = crowd_vals[i,0]
        carry_over_indices = np.append(carry_over_indices,index)

    carry_over_indices = np.asarray(carry_over_indices,dtype='int')

    of_array = np.copyto(of_array,double_of_array[carry_over_indices,:])
    player_vals = np.copyto(player_vals, new_array[carry_over_indices,:])

def plot_curves(p1):
    global nx, kxval

    ap3 = -p1[0]
    am3 = -p1[1]

    a0 = -10.0 * (am3 + ap3)
    ap1 = 5.0 * am3 + 10.0 * ap3 - 2.0 / 3.0
    ap2 = -am3 - 5.0 * ap3 + 1.0 / 12.0
    am1 = 10.0 * am3 + 5.0 * ap3 + 2.0 / 3.0
    am2 = -5.0 * am3 - ap3 - 1.0 / 12.0

    a0 = -a0
    ap1 = -ap1
    ap2 = -ap2
    ap3 = -ap3
    am1 = -am1
    am2 = -am2
    am3 = -am3

    im = complex(0, 1)
    k_x = np.zeros(nx + 1, dtype='double')

    start_index = 0
    for i in range(0, nx + 1):
        k_x[i] = np.pi * i / float(nx)
        if k_x[i] < kxval:
            start_index = start_index + 1

    mwe_complex = am3 * (np.cos(3.0 * k_x) - im * np.sin(3.0 * k_x)) + am2 * (
        np.cos(2.0 * k_x) - im * np.sin(2.0 * k_x)) + am1 * (np.cos(k_x) - im * np.sin(k_x))

    mwe_complex = mwe_complex + a0 + ap1 * (np.cos(k_x) + im * np.sin(k_x)) + ap2 * (
        np.cos(2.0 * k_x) + im * np.sin(2.0 * k_x)) + ap3 * (
        np.cos(3.0 * k_x) + im * np.sin(3.0 * k_x))

    dispersion = np.zeros((nx + 1), dtype='double')
    dissipation = np.zeros((nx + 1), dtype='double')

    for i in range(0, nx + 1):
        dissipation[i] = np.real(mwe_complex[i])
        dispersion[i] = np.imag(mwe_complex[i])


    plt.figure(1)
    plt.plot(k_x, dispersion)
    plt.title('Dispersion Curve')
    plt.xlabel('k')
    plt.ylabel('k*')

    plt.figure(2)
    plt.plot(k_x, dissipation)
    plt.title('Dissipation Curve')
    plt.xlabel('k')
    plt.ylabel('Dissipation')

    plt.show()

def plot_curves_comparison(p1,p2):
    global nx, kxval

    im = complex(0, 1)
    k_x = np.zeros(nx + 1, dtype='double')

    ap3 = -p1[0]
    am3 = -p1[1]

    a0 = -10.0 * (am3 + ap3)
    ap1 = 5.0*am3 + 10.0*ap3 - 2.0 / 3.0
    ap2 = -am3 - 5.0*ap3 + 1.0 / 12.0
    am1 = 10.0*am3+5.0*ap3+2.0/3.0
    am2 = -5.0*am3-ap3-1.0/12.0

    a0 = -a0
    ap1 = -ap1
    ap2 = -ap2
    ap3 = -ap3
    am1 = -am1
    am2 = -am2
    am3 = -am3

    im = complex(0, 1)
    k_x = np.zeros(nx + 1, dtype='double')

    start_index = 0
    for i in range(0, nx + 1):
        k_x[i] = np.pi * i / float(nx)
        if k_x[i] < kxval:
            start_index = start_index + 1

    mwe_complex = am3 * (np.cos(3.0 * k_x) - im * np.sin(3.0 * k_x)) + am2 * (
        np.cos(2.0 * k_x) - im * np.sin(2.0 * k_x)) + am1 * (np.cos(k_x) - im * np.sin(k_x))

    mwe_complex = mwe_complex + a0 + ap1 * (np.cos(k_x) + im * np.sin(k_x)) + ap2 * (
        np.cos(2.0 * k_x) + im * np.sin(2.0 * k_x)) + ap3 * (
        np.cos(3.0 * k_x) + im * np.sin(3.0 * k_x))

    dispersion = np.zeros((nx + 1), dtype='double')
    dissipation = np.zeros((nx + 1), dtype='double')

    for i in range(0, nx + 1):
        dissipation[i] = np.real(mwe_complex[i])
        dispersion[i] = np.imag(mwe_complex[i])

    plt.figure(1)
    plt.plot(k_x, dispersion,label='p1')
    plt.plot(k_x, k_x, label='spectral')
    plt.title('Dispersion Curve')
    plt.xlabel('k')
    plt.ylabel('k*')

    plt.figure(2)
    plt.plot(k_x, dissipation, label='p1')
    plt.title('Dissipation Curve')
    plt.xlabel('k')
    plt.ylabel('Dissipation')

    ap3 = -p2[0]
    am3 = -p2[1]

    a0 = -10.0 * (am3 + ap3)
    ap1 = 5.0 * am3 + 10.0 * ap3 - 2.0 / 3.0
    ap2 = -am3 - 5.0 * ap3 + 1.0 / 12.0
    am1 = 10.0 * am3 + 5.0 * ap3 + 2.0 / 3.0
    am2 = -5.0 * am3 - ap3 - 1.0 / 12.0

    a0 = -a0
    ap1 = -ap1
    ap2 = -ap2
    ap3 = -ap3
    am1 = -am1
    am2 = -am2
    am3 = -am3

    im = complex(0, 1)
    k_x = np.zeros(nx + 1, dtype='double')

    start_index = 0
    for i in range(0, nx + 1):
        k_x[i] = np.pi * i / float(nx)
        if k_x[i] < kxval:
            start_index = start_index + 1

    mwe_complex = am3 * (np.cos(3.0 * k_x) - im * np.sin(3.0 * k_x)) + am2 * (
        np.cos(2.0 * k_x) - im * np.sin(2.0 * k_x)) + am1 * (np.cos(k_x) - im * np.sin(k_x))

    mwe_complex = mwe_complex + a0 + ap1 * (np.cos(k_x) + im * np.sin(k_x)) + ap2 * (
        np.cos(2.0 * k_x) + im * np.sin(2.0 * k_x)) + ap3 * (
        np.cos(3.0 * k_x) + im * np.sin(3.0 * k_x))

    dispersion = np.zeros((nx + 1), dtype='double')
    dissipation = np.zeros((nx + 1), dtype='double')

    for i in range(0, nx + 1):
        dissipation[i] = np.real(mwe_complex[i])
        dispersion[i] = np.imag(mwe_complex[i])

    plt.figure(1)
    plt.plot(k_x, dispersion, label='p2')
    plt.legend()

    plt.figure(2)
    plt.plot(k_x, dissipation,label='p2')
    plt.legend()

    plt.show()

if __name__ == "__main__":

    nx = 128
    kxval = 0.0
    nplayers = 50
    sbx_param_n = 3
    mut_rate = 0.1
    tol = 1.0e-8
    max_dispersion_deviation = -0.0105864086319 #Max overshoot of curve over spectral line
    max_dissipation = 0.4 #Maximum value of dissipation

    plot_curves_comparison([1.0/60.0,-1.0/60.0],[0.02651995,-0.02651995])
    plot_curves_comparison([0.02651995, -0.02651995], [0.02495745, -0.02808245])

    print scheme_objective_function(0.02651995,-0.02651995)
    print scheme_objective_function(0.02495745,-0.02808245)

    of_array, player_vals = player_initialize(nplayers,nx)

    for i in range(1,50):
        print i
        nsga(player_vals, of_array)

        if i%5==0:
            crowd1_vals = crowding_sort(fast_non_dominated_sort(nplayers, of_array)[0].data, of_array)
            front1vals = fast_non_dominated_sort(nplayers, of_array)[0].data
            plot_front(1, of_array, front1vals, crowd1_vals)
            p1 = player_vals[front1vals[0], :]
            plot_curves_comparison(p1, [1.0/60.0, -1.0/60.0])
            print scheme_objective_function(p1[0], p1[1])
            print p1

    ap3 = -p1[0]
    am3 = -p1[1]

    a0 = -10.0 * (am3 + ap3)
    ap1 = 5.0 * am3 + 10.0 * ap3 - 2.0 / 3.0
    ap2 = -am3 - 5.0 * ap3 + 1.0 / 12.0
    am1 = 10.0 * am3 + 5.0 * ap3 + 2.0 / 3.0
    am2 = -5.0 * am3 - ap3 - 1.0 / 12.0

    a0 = -a0
    ap1 = -ap1
    ap2 = -ap2
    ap3 = -ap3
    am1 = -am1
    am2 = -am2
    am3 = -am3

    print am3, am2, am1, a0, ap1, ap2, ap3
