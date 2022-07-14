# import torch
# import torch.nn as nn
import numpy as np
import copy
import math
import time
import scipy as sc
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# 8
# 16   0.058873988765826946   0.05302928754551495
# 32   0.013857823346156595   0.012950746721879458
# 64   0.003218964440079519   0.003218964440079519
# 128

x = [0, 4]
y = [0, 4]

num = 32
Num_x = num
Num_y = num

h_x = (x[1]-x[0]) / Num_x
h_y = (y[1]-y[0]) / Num_y

X = np.linspace(x[0], x[1], Num_x + 1)
Y = np.linspace(y[0], y[1], Num_y + 1)

X = X[1:]
Y = Y[1:]

mesh = np.meshgrid(X, Y)
nodes = np.array(list(zip(*(dim.flat for dim in mesh))))
nodes = nodes[:, [1, 0]]
Num_all = nodes.shape[0]


id_all = list(range(Num_all))
id_bounary_corner = list([0, Num_x-1, Num_all-1-(Num_x-1), Num_all-1])
id_bounary_time_initial = list(set(list(np.where(nodes[:, 0] == x[0]+h_x)[0])).difference(id_bounary_corner))
id_bounary_time_end = list(set(list(np.where(nodes[:, 0] == x[1])[0])).difference(id_bounary_corner))
id_bounary_time = list(set(id_bounary_time_initial).union(id_bounary_time_end))
id_bounary_space = list(set(list(np.where(np.logical_or(nodes[:, 1] == y[0]+h_y, nodes[:, 1] == y[1]))[0])).difference(id_bounary_corner))
id_external = list(set(list(set(id_bounary_time).union(id_bounary_space))).union(id_bounary_corner))
id_internal = list(set(id_all).difference(id_external))


A = sc.sparse.lil_matrix((Num_all, Num_all))
for i in range(len(id_internal)):
    A[id_internal[i], id_internal[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
    A[id_internal[i], id_internal[i] - 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] + 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] - ((Num_x-1) + 1)] = -1 / (h_y * h_y)
    A[id_internal[i], id_internal[i] + ((Num_x-1) + 1)] = -1 / (h_y * h_y)

for i in range(len(id_external)):
    A[id_external[i], id_external[i]] = 1


for i in range(len(id_bounary_time_initial)):
    A[id_bounary_time_initial[i], id_bounary_time_initial[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
    A[id_bounary_time_initial[i], id_bounary_time_initial[i] + ((Num_x-1) + 1)] = -1 / (h_y * h_y)
    A[id_bounary_time_initial[i], id_bounary_time_initial[i] + ((Num_x-1) + 1)*(Num_y-1)] = -1 / (h_y * h_y)
    A[id_bounary_time_initial[i], id_bounary_time_initial[i] - 1] = -1 / (h_x * h_x)
    A[id_bounary_time_initial[i], id_bounary_time_initial[i] + 1] = -1 / (h_x * h_x)

for i in range(len(id_bounary_time_end)):
    A[id_bounary_time_end[i], id_bounary_time_end[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
    A[id_bounary_time_end[i], id_bounary_time_end[i] - ((Num_x-1) + 1)] = -1 / (h_y * h_y)
    A[id_bounary_time_end[i], id_bounary_time_end[i] - ((Num_x-1) + 1)*(Num_y-1)] = -1 / (h_y * h_y)
    A[id_bounary_time_end[i], id_bounary_time_end[i] - 1] = -1 / (h_x * h_x)
    A[id_bounary_time_end[i], id_bounary_time_end[i] + 1] = -1 / (h_x * h_x)



for i in range(len(id_bounary_space)):
    cur_nodes = nodes[id_bounary_space[i], :]
    A[id_bounary_space[i], id_bounary_space[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
    if cur_nodes[1] == y[0]+h_y:
        A[id_bounary_space[i], id_bounary_space[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x-1)] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + ((Num_x-1) + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - ((Num_x-1) + 1)] = -1 / (h_y * h_y)
    elif cur_nodes[1] == y[1]:
        A[id_bounary_space[i], id_bounary_space[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x-1)] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + ((Num_x-1) + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - ((Num_x-1) + 1)] = -1 / (h_y * h_y)



for i in range(len(id_bounary_corner)):
    A[id_bounary_corner[i], id_bounary_corner[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
    if i == 0:
        A[id_bounary_corner[i], id_bounary_corner[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x-1)] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] + ((Num_x-1) + 1)] = -1 / (h_y * h_y)
        A[id_bounary_corner[i], id_bounary_corner[i] + ((Num_x-1) + 1) * (Num_y-1)] = -1 / (h_y * h_y)
    elif i == 1:
        A[id_bounary_corner[i], id_bounary_corner[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x-1)] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] + ((Num_x-1) + 1)] = -1 / (h_y * h_y)
        A[id_bounary_corner[i], id_bounary_corner[i] + ((Num_x-1) + 1)*(Num_y-1)] = -1 / (h_y * h_y)
    elif i == 2:
        A[id_bounary_corner[i], id_bounary_corner[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x-1)] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] - ((Num_x-1) + 1)] = -1 / (h_y * h_y)
        A[id_bounary_corner[i], id_bounary_corner[i] - ((Num_x-1) + 1)*(Num_y-1)] = -1 / (h_y * h_y)
    elif i == 3:
        A[id_bounary_corner[i], id_bounary_corner[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x-1)] = -1 / (h_x * h_x)
        A[id_bounary_corner[i], id_bounary_corner[i] - ((Num_x-1) + 1)] = -1 / (h_y * h_y)
        A[id_bounary_corner[i], id_bounary_corner[i] - ((Num_x-1) + 1)*(Num_y-1)] = -1 / (h_y * h_y)



def f(x, y):
    return 2*((np.pi)**2) * np.cos(np.pi*x)*np.cos(np.pi*y)   #2*((np.pi)**2) * np.sin(np.pi*x)*np.sin(np.pi*y)  #


def real_u(nodes):
    x = nodes[:, 0]
    y = nodes[:, 1]
    return np.cos(np.pi*x)*np.cos(np.pi*y)   #np.sin(np.pi*x)*np.sin(np.pi*y)  #



b = f(nodes[:, 0], nodes[:, 1])

b0 = np.ones(Num_all)

A_plus = sc.sparse.lil_matrix((Num_all+1, Num_all+1))
A_plus[0:Num_all, 0:Num_all] = A
A_plus[0:Num_all, Num_all] = h_x*h_y*b0
A_plus[Num_all, 0:Num_all] = h_x*h_y*b0.T

b_plus = np.hstack((b, [0]))


from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
cond_A = np.linalg.cond(A.toarray())
print(cond_A)

u_h = cg(A_plus, b_plus)[0][0:Num_all]
# u_h = cg(A, b)[0]
u_r = real_u(nodes)

error_inf = np.abs((u_h - u_r)).max()
print(error_inf)




from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(nodes[:, 0], nodes[:, 1], abs(u_h-u_r), c='r')
ax.scatter(nodes[:, 0], nodes[:, 1], u_h, c='r')
ax.scatter(nodes[:, 0], nodes[:, 1], u_r, c='b')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
ww = 1
