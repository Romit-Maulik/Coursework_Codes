#Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
#email:romit.maulik@okstate.edu
#09-17-2017
import numpy as np
import matplotlib.pyplot as plt

y = np.random.rand(10)
x = np.linspace(0,9,10)

plt.figure(1)
plt.interactive(False)
plt.xlabel('x')
plt.ylabel('y')
plt.title('My First Scatter')
plt.scatter(x, y)
plt.show()