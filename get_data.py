import numpy as np
from xy import *
import pickle, pprint
import matplotlib.pyplot as plt
import math


#Parameters might change
J = 1
max_t = 2.05
min_t = 0.05
lattice_shape = (16,16) #can be changed to (16,16) or (32,32)
steps = 1
iters_per_step = 80
random_state = 30
t_vals = np.linspace(min_t, max_t, 32)
print(t_vals)
#n = 2
#t_vals = np.concatenate([t_vals[:13-n],t_vals[13+n:]])
# betas = 1 / T_vals
lattices = []
#Monte Carlo Simulation
for beta in t_vals:
        lat=[]
        print(beta)
        random_state=random_state+1
        xy=XYModelMetropolisSimulation(lattice_shape=lattice_shape,beta=1/beta,J=J,random_state=random_state)
        for q in range(150):
            xy.simulate(steps,iters_per_step)
            lat.append(xy.L+0)
            # draw_grid(lattice_shape[0],xy.L,1/beta)
        lattices.append(lat[120:])
        print('Done')
#Saving Data
output = open(str(lattice_shape)+'lattices.pkl', 'wb')
pickle.dump(lattices, output)
output.close()
