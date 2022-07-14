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
# 16   0.05356142717549783       0.05569929051461808
# 32   0.013066625310367064      0.01353419028543068
# 64   0.0032469139184543305      0.003359740038865189
# 128

x = [0, 4]
y = [0, 4]

num = 64
Num_x = num
Num_y = num

h_x = (x[1] - x[0]) / Num_x
h_y = (y[1] - y[0]) / Num_y

X = np.linspace(x[0], x[1], Num_x + 1)
Y = np.linspace(y[0], y[1], Num_y + 1)

mesh = np.meshgrid(X, Y)
nodes = np.array(list(zip(*(dim.flat for dim in mesh))))
nodes = nodes[:, [1, 0]]
Num_all = nodes.shape[0]


id_all = list(range(Num_all))
id_bounary_corner = list([0, Num_x, Num_all-1-Num_x, Num_all-1])
id_bounary_time_initial = list(set(list(np.where(nodes[:, 0] == x[0])[0])).difference(id_bounary_corner))
id_bounary_time_end = list(set(list(np.where(nodes[:, 0] == x[1])[0])).difference(id_bounary_corner))
id_bounary_time = list(set(id_bounary_time_initial).union(id_bounary_time_end))
id_bounary_space = list(set(list(np.where(np.logical_or(nodes[:, 1] == y[0], nodes[:, 1] == y[1]))[0])).difference(id_bounary_corner))
id_external = list(set(list(set(id_bounary_time).union(id_bounary_space))).union(id_bounary_corner))
id_internal = list(set(id_all).difference(id_external))


A = sc.sparse.lil_matrix((Num_all, Num_all))
for i in range(len(id_internal)):
    A[id_internal[i], id_internal[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
    A[id_internal[i], id_internal[i] - 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] + 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] - (Num_x + 1)] = -1 / (h_y * h_y)
    A[id_internal[i], id_internal[i] + (Num_x + 1)] = -1 / (h_y * h_y)

for i in range(len(id_external)):
    A[id_external[i], id_external[i]] = 1


# for i in range(len(id_bounary_time_initial)):
#     A[id_bounary_time_initial[i], id_bounary_time_initial[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
#     A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)] = -2 / (h_y * h_y)
#     A[id_bounary_time_initial[i], id_bounary_time_initial[i] - 1] = -1 / (h_x * h_x)
#     A[id_bounary_time_initial[i], id_bounary_time_initial[i] + 1] = -1 / (h_x * h_x)
#
# for i in range(len(id_bounary_time_end)):
#     A[id_bounary_time_end[i], id_bounary_time_end[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
#     A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)] = -2 / (h_y * h_y)
#     A[id_bounary_time_end[i], id_bounary_time_end[i] - 1] = -1 / (h_x * h_x)
#     A[id_bounary_time_end[i], id_bounary_time_end[i] + 1] = -1 / (h_x * h_x)

for i in range(len(id_bounary_space)):
    cur_nodes = nodes[id_bounary_space[i], :]
    if cur_nodes[1] == y[0]:
        A[id_bounary_space[i], id_bounary_space[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] + 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1)] = -1 / (h_y * h_y)
    elif cur_nodes[1] == y[1]:
        A[id_bounary_space[i], id_bounary_space[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1)] = -1 / (h_y * h_y)


def f(x, y):
    return 2*((np.pi)**2) * np.sin(np.pi*x)*np.sin(np.pi*y)  #2*((np.pi)**2) * np.cos(np.pi*x)*np.cos(np.pi*y)   #


def real_u(nodes):
    x = nodes[:, 0]
    y = nodes[:, 1]
    return np.sin(np.pi*x)*np.sin(np.pi*y)  #np.cos(np.pi*x)*np.cos(np.pi*y)   #


def N_bounary(nodes):
    out = np.zeros((nodes.shape[0]))
    for i in range(nodes.shape[0]):
        cur_x = nodes[i, 0]
        cur_y = nodes[i, 1]
        temp = nodes[i]
        if temp[1] == y[0]:
            out[i] = f(cur_x, cur_y) - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)) / h_y
            # out[i] = (f(cur_x, cur_y) - 2 * (-np.pi * np.cos(np.pi * cur_x) * np.sin(np.pi * cur_y)) / h_y)
        elif temp[1] == y[1]:
            out[i] = f(cur_x, cur_y) + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)) / h_y
            # out[i] = (f(cur_x, cur_y) + 2 * (-np.pi * np.cos(np.pi * cur_x) * np.sin(np.pi * cur_y)) / h_y)
        elif temp[0] == x[0] and temp[1] != y[0] and temp[1] != y[1]:
            out[i] = f(cur_x, cur_y) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)) / h_x
            # out[i] = (f(cur_x, cur_y) - 2 * (-np.pi * np.sin(np.pi * cur_x) * np.cos(np.pi * cur_y)) / h_x)
        elif temp[0] == x[1] and temp[1] != y[0] and temp[1] != y[1]:
            out[i] = f(cur_x, cur_y) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)) / h_x
            # out[i] = (f(cur_x, cur_y) + 2 * (-np.pi * np.sin(np.pi * cur_x) * np.cos(np.pi * cur_y)) / h_x)
    return out


b = f(nodes[:, 0], nodes[:, 1])
b[id_bounary_corner] = real_u(nodes[id_bounary_corner, :])

b[id_bounary_space] = N_bounary(nodes[id_bounary_space, :])
b[id_bounary_time_initial] = real_u(nodes[id_bounary_time_initial, :])
b[id_bounary_time_end] = real_u(nodes[id_bounary_time_end, :])



from scipy.sparse.linalg import spsolve, cg
cond_A = np.linalg.cond(A.toarray())
print(cond_A)

# u_h = cg(A_plus, b_plus)[0][0:Num_all]
u_h = spsolve(A, b)
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
