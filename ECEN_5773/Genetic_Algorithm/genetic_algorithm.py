import numpy as np
import matplotlib.pyplot as plt
import random

# Genetic programming code to minimize single-valued objective function
def objective_function(array):
    of = 100.0 * (array[:, 0] ** 2 - array[:, 1]) ** 2 + (1.0 - array[:, 0]) ** 2 #Rosenbock
    return of

def single_objective_function(array):
    of = 100.0*(array[0] ** 2 - array[1])**2 + (1.0-array[0])**2
    return of

def ofcalc(a, b):
    b = np.copyto(b,objective_function(a))

def genetic_algorithm_optimization(players,of_array):
    #Initial best solution location and value
    minpos = np.argmin(of_array)
    best_sol = np.copy(players[minpos])

    global nplayers,chi,std_conv

    global_iter = 0
    dist_metric = np.array([[0, single_objective_function(best_sol)]])

    while np.std(of_array)>std_conv:

        global_iter = global_iter + 1
        #Copy operation
        minval_indices = int(chi*nplayers)
        if minval_indices%2 == 1:#Make stuff even
            minval_indices = minval_indices + 1

        #Copying the best players to next generation
        new_players = np.copy(players[np.argpartition(of_array,minval_indices)[:minval_indices]])

        # Crossover of remaining players
        global sbx_param_n
        crossover_pairs = np.copy(players[np.argpartition(of_array, minval_indices)[:minval_indices]])
        # Time to pair them up - already random - crossover in place
        for i in range(0, nplayers - minval_indices, 2):
            p1 = crossover_pairs[i,:]
            p2 = crossover_pairs[i + 1,:]
            u = np.random.uniform(0.0, 1.0)

            beta = 0.0

            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (sbx_param_n + 1))
            else:
                beta = (1.0 / (2.0 - 2.0 * u)) ** (1.0 / (sbx_param_n + 1))

            c1 = 0.5 * (p1 + p2) - 0.5 * beta * (p2 - p1)
            c2 = 0.5 * (p1 + p2) + 0.5 * beta * (p2 - p1)

            crossover_pairs[i,:] = c1
            crossover_pairs[i + 1,:] = c2

        #Concatenate randomly
        cp_new = np.random.randint(np.shape(crossover_pairs)[0],size=nplayers-minval_indices)
        crossover_pairs = crossover_pairs[cp_new,:]

        np.copyto(players,np.concatenate((new_players,crossover_pairs),axis=0))

        #Mutation of new generation
        global mut_rate
        mut_indices = random.sample(range(0, nplayers), int(mut_rate * nplayers))#Generate random numbers which are not duplicate
        players[mut_indices[:]] = players[mut_indices[:]]*np.random.uniform(-1.0,1.0)

        #Update objective function array
        ofcalc(players,of_array)
        minpos = np.argmin(of_array)
        new_best_sol = players[minpos]

        if single_objective_function(new_best_sol)<single_objective_function(best_sol):
            np.copyto(best_sol,new_best_sol)
            newrow = np.array([[global_iter, single_objective_function(best_sol)]])
            dist_metric = np.concatenate((dist_metric, newrow), axis=0)
            del newrow

        del crossover_pairs,new_players,mut_indices,new_best_sol

    print('The best parameters are: ',best_sol)
    print('The best value is: ', single_objective_function(best_sol))

    #Plotting progress to convergence
    plt.figure(1)
    plt.interactive(False)
    plt.title('Progress to convergence - Genetic Algorithm')
    plt.xlabel('Iterations')
    plt.plot(dist_metric[:,0],dist_metric[:,1])
    plt.yscale('log')
    plt.plot()
    plt.show()

    #Plotting contours
    x = np.arange(-2.048,2.048,0.1)
    y = np.arange(-2.048, 2.048, 0.1)
    xx, yy = np.meshgrid(x,y,sparse=False)
    z = single_objective_function([xx,yy])
    h = plt.contourf(x,y,z)
    plt.title("Minima found")
    plt.plot(best_sol[0],best_sol[1], 'ro')
    plt.colorbar(h,format="%.2f")
    plt.show()

if __name__ == "__main__":
    nplayers = 400           #Division by two should be even number
    nparam = 2
    lower_limit = -2.048
    upper_limit = 2.048
    sbx_param_n = 3
    players = np.random.uniform(low=lower_limit, high=upper_limit, size=(nplayers, nparam))
    # Initialize array with random numbers within limits
    of_array = np.zeros((nplayers,))
    # Calculate initial objective function values of all players
    ofcalc(players, of_array)

    #Time for GP optimization
    chi = 0.5 #Fraction of population to be copied to new generation
    mut_rate = 0.2 #Fraction of population that mutates
    std_conv = 1.0e-3
    genetic_algorithm_optimization(players,of_array)
