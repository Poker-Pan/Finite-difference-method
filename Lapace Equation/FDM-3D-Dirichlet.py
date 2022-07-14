import numpy as np
import copy
import math
import time
import scipy as sc
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



# 8    0.23370055013617108     0.2605941890701291
# 16   0.05302928754551672     0.056520090196042316
# 32   0.012950746721884343    0.013661259671079451



x = [0, 4]
y = [0, 4]
z = [0, 4]

num = 16
Num_x = num
Num_y = num
Num_z = num

h_x = (x[1] - x[0]) / Num_x
h_y = (y[1] - y[0]) / Num_y
h_z = (z[1] - z[0]) / Num_z

X = np.linspace(x[0], x[1], Num_x + 1)
Y = np.linspace(y[0], y[1], Num_y + 1)
Z = np.linspace(z[0], z[1], Num_z + 1)

mesh = np.meshgrid(X, Y, Z)
nodes = np.array(list(zip(*(dim.flat for dim in mesh))))
nodes = nodes[:, [1, 2, 0]]  # t, x, y
Num_all = nodes.shape[0]

id_all = list(range(Num_all))
id_bounary_time_initial = list(np.where(nodes[:, 0] == x[0])[0])
id_bounary_time_end = list(np.where(nodes[:, 0] == x[1])[0])
id_bounary_time = list(set(id_bounary_time_initial).union(id_bounary_time_end))
id_bounary_space = list(set(list(np.where(np.logical_or(np.logical_or(nodes[:, 1] == y[0], nodes[:, 1] == y[1]),
                                                        np.logical_or(nodes[:, 2] == z[0], nodes[:, 2] == z[1])))[0])).difference(id_bounary_time))
id_external = list(set(id_bounary_time).union(id_bounary_space))
id_internal = list(set(id_all).difference(id_external))


A = sc.sparse.lil_matrix((Num_all, Num_all))
for i in range(len(id_internal)):
    A[id_internal[i], id_internal[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y) + 2 / (h_z * h_z)
    A[id_internal[i], id_internal[i] - 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] + 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] - (Num_x + 1)] = -1 / (h_y * h_y)
    A[id_internal[i], id_internal[i] + (Num_x + 1)] = -1 / (h_y * h_y)
    A[id_internal[i], id_internal[i] - (Num_x + 1)*(Num_y + 1)] = -1 / (h_z * h_z)
    A[id_internal[i], id_internal[i] + (Num_x + 1)*(Num_y + 1)] = -1 / (h_z * h_z)

for i in range(len(id_external)):
    A[id_external[i], id_external[i]] = 1

def f(x, y, z):
    # return 3*((np.pi)**2) * np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    return 3*((np.pi)**2) * np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)

def real_u(nodes):
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]
    # return np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
    return np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)

b = f(nodes[:, 0], nodes[:, 1], nodes[:, 2])
b[id_external] = real_u(nodes[id_external, :])

from scipy.sparse.linalg import spsolve, cg, lsqr
cond_A = np.linalg.cond(A.toarray())
print(cond_A)

u_h = spsolve(A, b)
u_r = real_u(nodes)

error_inf = np.abs((u_h - u_r)).max()
print(error_inf)



































