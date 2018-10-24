import numpy as np
import xlrd
import random
import matplotlib.pyplot as plt
import os


# Import city data from excel file
book = xlrd.open_workbook('C:\Users\Students\Desktop\Fall_2017\MAE_5773\Homework2\Problem1\City_Data.xlsx')
sheet = book.sheet_by_name('Sheet1')
data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]

# Convert this to a numpy array
city_array = np.asarray(data)

# Calculate number of cities
num_cities = city_array.shape[0]
print("The number of cities being solver for are", num_cities)

# Defining a distance function
def calc_distance(array):
    dist = 0.0
    for i in range(1, num_cities):
        dist = dist + ((array[i, 1] - array[i - 1, 1]) ** (2) + (array[i, 2] - array[i - 1, 2]) ** (2)) ** (1 / 2.0)

    dist = dist + ((array[1, 1] - array[num_cities - 1, 1]) ** (2) + (array[1, 2] - array[num_cities - 1, 2]) ** (
    2)) ** (1 / 2.0)
    return dist


# Calculating initial random distance
print('The initial random distance is ', calc_distance(city_array))


# Point to point reversal perturbation
def perturb_b(array, temp_array):
    i1, i2 = random.sample(range(1, num_cities - 1), 2)#Generate random numbers which are duplicate
    temp_array[i1] = array[i2]
    temp_array[i2] = array[i1]

    if i1>i2:
        j = i2
        for i in range(i1,i2+1):
            temp_array[i] = array[j]
            array[j] = temp_array[i]
            j = j-1
    else:
        j = i1
        for i in range(i2, i1 + 1):
            array[i] = temp_array[j]
            temp_array[j] = array[i]
            j = j - 1

#Point to point perturbation
def perturb_a(array, temp_array):
    i1, i2 = random.sample(range(1, num_cities - 1), 2)  # Generate random numbers which are not duplicate
    temp_array[i1] = array[i2]
    temp_array[i2] = array[i1]

# Define function for simulated annealing
max_iter = 1000        # First fixing a max number of iterations
num_restarts = 100    # Number of reinitializations
temperature = 1.0e8  # Initial temperature
k = 1                 # Boltzmann Constant
alpha = 0.995          # Temperature variation parameter
conv_temp = 0.0001     # Stopping criteria


def simulated_annealing_tsp(array):

    original_array = np.copy(array)
    # Random shuffling to select first solution
    np.random.shuffle(array[1:, :])

    best_array = np.copy(array)
    best_sol = calc_distance(best_array)
    global_iter = 0

    for restarts in range(1,num_restarts):
        for iter in range(0,max_iter):
            global_iter = global_iter+1
            # Global modification of temperature
            global temperature
            global dist_metric

            # Perturb according to scheme
            temp_array = np.copy(array)  # New numpy array - NOT REFERENCE
            perturb_b(array, temp_array)

            # Check feasibility of new solutions
            old_sol = calc_distance(array)
            new_sol = calc_distance(temp_array)

            if new_sol <= old_sol and new_sol <= best_sol:
                np.copyto(array, temp_array)
                np.copyto(best_array, temp_array)
                print('Better Solution ', new_sol)
                mysol = new_sol
                best_sol = new_sol
            elif (new_sol <= old_sol):
                np.copyto(array, temp_array)
                print('Better Solution ', new_sol)
                mysol = new_sol
            else:
                prob = np.exp(-(new_sol - old_sol) / (k * temperature))
                rand_num = np.random.uniform(0.0, 1.0)
                if prob > rand_num:
                    np.copyto(array, temp_array)
                    print('Worse solution accepted')
                    mysol = new_sol
                else:
                    print('Old solution')
                    mysol = old_sol

            temperature = alpha * temperature

            # Check for convergence
            if temperature < conv_temp:
                np.random.shuffle(original_array[1:, :])
                array = np.copy(original_array)
                temperature = 1.0e8

            print calc_distance(best_array)
            del temp_array
            newrow = np.array([[global_iter, mysol]])
            dist_metric = np.concatenate((dist_metric, newrow),axis=0)

    # Output final answer
    print('The smallest found distance is', calc_distance(best_array))
    np.copyto(city_array,best_array)
    del best_array

dist_metric = np.array([[0,calc_distance(city_array)]])
simulated_annealing_tsp(city_array)

# Plot our scatter
plt.figure(1)
plt.interactive(False)
plt.title('20 City TSP Solution - Simulated Annealing')
plt.xlabel('x')
plt.ylabel('y')

#Final connector
newrow = city_array[0]
city_array = np.vstack([city_array, newrow])

plt.plot(city_array[:, 1], city_array[:, 2],'.r-')

for i in range (0,num_cities):
    xy=(city_array[i,1],city_array[i,2])
    plt.annotate(int(city_array[i,0]),xy)

plt.plot()

plt.figure(2)
plt.interactive(False)
plt.title('20 City TSP Progress to Solution - Simulated Annealing')
plt.xlabel('Iterations')
plt.plot(dist_metric[:, 0], dist_metric[:, 1])
plt.plot()



plt.show()
