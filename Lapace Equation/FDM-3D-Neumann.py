import numpy as np
import copy
import math
import time
import scipy as sc
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



# 8   0.23370055013617042    0.23974303255778182
# 16  0.05302928754551561    0.08525670873949126
# 32  0.012950746721881234   0.023143350795511358
# 48                         0.010470879763682876
# 64                         0.005925121815985258

x = [0.0, 1.0]
y = [0.0, 1.0]
z = [0.0, 1.0]

num = 8
Num_x = num
Num_y = num
Num_z = num

h_x = (x[1] - x[0]) / Num_x
h_y = (y[1] - y[0]) / Num_y
h_z = (z[1] - z[0]) / Num_z

X = np.linspace(x[0], x[1], Num_x + 1)
Y = np.linspace(y[0], y[1], Num_y + 1)
Z = np.linspace(z[0], z[1], Num_z + 1)

mesh = np.meshgrid(Z, X, Y)
nodes = np.array(list(zip(*(dim.flat for dim in mesh))))
nodes = nodes[:, [1, 2, 0]]  # t, x, y
Num_all = nodes.shape[0]




id_all = list(range(Num_all))
id_bounary_corner = list([0, Num_x, ((Num_x+1)*(Num_y+1)-1) - Num_x, (Num_x+1)*(Num_y+1)-1, ((Num_x+1)*(Num_y+1)*(Num_z+1)-1) - ((Num_x+1)*(Num_y+1)-1), ((Num_x+1)*(Num_y+1)*(Num_z+1)-1) - ((Num_x+1)*(Num_y+1)-1) + Num_x, ((Num_x+1)*(Num_y+1)*(Num_z+1)-1) - Num_x, (Num_x+1)*(Num_y+1)*(Num_z+1)-1])
id_bounary_time_initial = list(set(list(np.where(nodes[:, 0] == x[0])[0])).difference(id_bounary_corner))
id_bounary_time_end = list(set(list(np.where(nodes[:, 0] == x[1])[0])).difference(id_bounary_corner))
id_bounary_time = list(set(id_bounary_time_initial).union(id_bounary_time_end))
id_bounary_space = list(set(list(set(list(np.where(np.logical_or(np.logical_or(nodes[:, 1] == y[0], nodes[:, 1] == y[1]),
                                                                 np.logical_or(nodes[:, 2] == z[0], nodes[:, 2] == z[1])))[0])).difference(id_bounary_time))).difference(id_bounary_corner))

id_external = list(set(list(set(id_bounary_time).union(id_bounary_space))).union(id_bounary_corner))
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


for i in range(len(id_bounary_time_initial)):
    cur_nodes = nodes[id_bounary_time_initial[i], :]
    A[id_bounary_time_initial[i], id_bounary_time_initial[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y) + 2 / (h_z * h_z)
    # 4 edge + 1 face
    if cur_nodes[1] == y[0]:
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + 1] = -2 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
    elif cur_nodes[1] == y[1]:
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - 1] = -2 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)
    elif cur_nodes[2] == z[0]:
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)
    elif cur_nodes[2] == z[1]:
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)
    else:
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_initial[i], id_bounary_time_initial[i] + (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)


for i in range(len(id_bounary_time_end)):
    cur_nodes = nodes[id_bounary_time_end[i], :]
    A[id_bounary_time_end[i], id_bounary_time_end[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y) + 2 / (h_z * h_z)
    # 4 edge + 1 face
    if cur_nodes[1] == y[0]:
        A[id_bounary_time_end[i], id_bounary_time_end[i] + 1] = -2 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)
    elif cur_nodes[1] == y[1]:
        A[id_bounary_time_end[i], id_bounary_time_end[i] - 1] = -2 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)
    elif cur_nodes[2] == z[0]:
        A[id_bounary_time_end[i], id_bounary_time_end[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] + (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)
    elif cur_nodes[2] == z[1]:
        A[id_bounary_time_end[i], id_bounary_time_end[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)
    else:
        A[id_bounary_time_end[i], id_bounary_time_end[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_time_end[i], id_bounary_time_end[i] - (Num_x + 1)*(Num_y + 1)] = -2 / (h_z * h_z)


for i in range(len(id_bounary_space)):
    cur_nodes = nodes[id_bounary_space[i], :]
    A[id_bounary_space[i], id_bounary_space[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y) + 2 / (h_z * h_z)
    # 4 face + 4 edge
    if cur_nodes[1] == y[0] and cur_nodes[2] != z[0] and cur_nodes[2] != z[1]:
        A[id_bounary_space[i], id_bounary_space[i] + 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
    elif cur_nodes[1] == y[1] and cur_nodes[2] != z[0] and cur_nodes[2] != z[1]:
        A[id_bounary_space[i], id_bounary_space[i] - 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1)] = -1 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
    elif cur_nodes[2] == z[0] and cur_nodes[1] != y[0] and cur_nodes[1] != y[1]:
        A[id_bounary_space[i], id_bounary_space[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
    elif cur_nodes[2] == z[1] and cur_nodes[1] != y[0] and cur_nodes[1] != y[1]:
        A[id_bounary_space[i], id_bounary_space[i] - 1] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + 1] = -1 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
    elif cur_nodes[1] == y[0] and cur_nodes[2] == z[0]:
        A[id_bounary_space[i], id_bounary_space[i] + 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
    elif cur_nodes[1] == y[1] and cur_nodes[2] == z[0]:
        A[id_bounary_space[i], id_bounary_space[i] - 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
    elif cur_nodes[1] == y[0] and cur_nodes[2] == z[1]:
        A[id_bounary_space[i], id_bounary_space[i] + 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
    elif cur_nodes[1] == y[1] and cur_nodes[2] == z[1]:
        A[id_bounary_space[i], id_bounary_space[i] - 1] = -2 / (h_x * h_x)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1)] = -2 / (h_y * h_y)
        A[id_bounary_space[i], id_bounary_space[i] - (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)
        A[id_bounary_space[i], id_bounary_space[i] + (Num_x + 1) * (Num_y + 1)] = -1 / (h_z * h_z)


for i in range(len(id_bounary_corner)):
    cur_nodes = nodes[id_bounary_corner[i], :]
    A[id_bounary_corner[i], id_bounary_corner[i]] = 2 / (h_x * h_x) + 2 / (h_y * h_y) + 2 / (h_z * h_z)
    if cur_nodes[0] == x[0]:
        if cur_nodes[1] == y[0] and cur_nodes[2] == z[0]:
            A[id_bounary_corner[i], id_bounary_corner[i] + 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
        elif cur_nodes[1] == y[1] and cur_nodes[2] == z[0]:
            A[id_bounary_corner[i], id_bounary_corner[i] - 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
        elif cur_nodes[1] == y[0] and cur_nodes[2] == z[1]:
            A[id_bounary_corner[i], id_bounary_corner[i] + 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
        elif cur_nodes[1] == y[1] and cur_nodes[2] == z[1]:
            A[id_bounary_corner[i], id_bounary_corner[i] - 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
    elif cur_nodes[0] == x[1]:
        if cur_nodes[1] == y[0] and cur_nodes[2] == z[0]:
            A[id_bounary_corner[i], id_bounary_corner[i] + 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
        elif cur_nodes[1] == y[1] and cur_nodes[2] == z[0]:
            A[id_bounary_corner[i], id_bounary_corner[i] - 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] + (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
        elif cur_nodes[1] == y[0] and cur_nodes[2] == z[1]:
            A[id_bounary_corner[i], id_bounary_corner[i] + 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)
        elif cur_nodes[1] == y[1] and cur_nodes[2] == z[1]:
            A[id_bounary_corner[i], id_bounary_corner[i] - 1] = -2 / (h_x * h_x)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1)] = -2 / (h_y * h_y)
            A[id_bounary_corner[i], id_bounary_corner[i] - (Num_x + 1) * (Num_y + 1)] = -2 / (h_z * h_z)


def f(x, y, z):
    return 3*((np.pi)**2) * np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)


def real_u(nodes):
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]
    return np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)


def N_bounary(nodes):
    out = np.zeros((nodes.shape[0]))
    for i in range(nodes.shape[0]):
        cur_x = nodes[i, 0]
        cur_y = nodes[i, 1]
        cur_z = nodes[i, 2]
        # 8 point
        if cur_x == x[0]:
            if cur_y == y[0] and cur_z == z[0]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[1] and cur_z == z[0]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[0] and cur_z == z[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[1] and cur_z == z[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
        elif cur_x == x[1]:
            if cur_y == y[0] and cur_z == z[0]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[1] and cur_z == z[0]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[0] and cur_z == z[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[1] and cur_z == z[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
        #  6 face +  12 edge
        if cur_x == x[0]:
            if cur_y == y[0] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x
            elif cur_y == y[1] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x
            elif cur_z == z[0] and cur_y != y[0] and cur_y != y[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_z == z[1] and cur_y != y[0] and cur_y != y[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y != y[0] and cur_y != y[1] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z
        elif cur_x == x[1]:
            if cur_y == y[0] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x
            elif cur_y == y[1] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x
            elif cur_z == z[0] and cur_y != y[0] and cur_y != y[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_z == z[1] and cur_y != y[0] and cur_y != y[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y != y[0] and cur_y != y[1] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.cos(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_z
        elif cur_x != x[0] and cur_x != x[1]:
            if cur_y == y[0] and cur_z == z[0]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[1] and cur_z == z[0]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[0] and cur_z == z[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[1] and cur_z == z[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_y == y[0] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x
            elif cur_y == y[1] and cur_z != z[0] and cur_z != z[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.sin(np.pi*cur_x)*np.cos(np.pi*cur_y)*np.sin(np.pi*cur_z))/h_x
            elif cur_z == z[0] and cur_y != y[0] and cur_y != y[1]:
                out[i] = f(cur_x, cur_y, cur_z) - 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
            elif cur_z == z[1] and cur_y != y[0] and cur_y != y[1]:
                out[i] = f(cur_x, cur_y, cur_z) + 2*(np.pi*np.sin(np.pi*cur_x)*np.sin(np.pi*cur_y)*np.cos(np.pi*cur_z))/h_y
    return out


b = f(nodes[:, 0], nodes[:, 1], nodes[:, 2])

b[id_bounary_corner] = N_bounary(nodes[id_bounary_corner, :])
b[id_bounary_time_initial] = N_bounary(nodes[id_bounary_time_initial, :])
b[id_bounary_time_end] = N_bounary(nodes[id_bounary_time_end, :])
b[id_bounary_space] = N_bounary(nodes[id_bounary_space, :])


b0 = np.ones(Num_all)
b0[id_bounary_corner] = 1/8
for i in range(len(id_bounary_time_initial)):
    cur_nodes = nodes[id_bounary_time_initial[i], :]
    if cur_nodes[1] == y[0]:
        b0[id_bounary_time_initial[i]] = 1/4
    elif cur_nodes[1] == y[1]:
        b0[id_bounary_time_initial[i]] = 1/4
    elif cur_nodes[2] == z[0]:
        b0[id_bounary_time_initial[i]] = 1/4
    elif cur_nodes[2] == z[1]:
        b0[id_bounary_time_initial[i]] = 1/4
    else:
        b0[id_bounary_time_initial[i]] = 1/2

for i in range(len(id_bounary_time_end)):
    cur_nodes = nodes[id_bounary_time_end[i], :]
    if cur_nodes[1] == y[0]:
        b0[id_bounary_time_end[i]] = 1/4
    elif cur_nodes[1] == y[1]:
        b0[id_bounary_time_end[i]] = 1/4
    elif cur_nodes[2] == z[0]:
        b0[id_bounary_time_end[i]] = 1/4
    elif cur_nodes[2] == z[1]:
        b0[id_bounary_time_end[i]] = 1/4
    else:
        b0[id_bounary_time_end[i]] = 1/2

for i in range(len(id_bounary_space)):
    cur_nodes = nodes[id_bounary_space[i], :]
    if cur_nodes[1] == y[0] and cur_nodes[2] != z[0] and cur_nodes[2] != z[1]:
        b0[id_bounary_space[i]] = 1/2
    elif cur_nodes[1] == y[1] and cur_nodes[2] != z[0] and cur_nodes[2] != z[1]:
        b0[id_bounary_space[i]] = 1/2
    elif cur_nodes[2] == z[0] and cur_nodes[1] != y[0] and cur_nodes[1] != y[1]:
        b0[id_bounary_space[i]] = 1/2
    elif cur_nodes[2] == z[1] and cur_nodes[1] != y[0] and cur_nodes[1] != y[1]:
        b0[id_bounary_space[i]] = 1/2
    elif cur_nodes[1] == y[0] and cur_nodes[2] == z[0]:
        b0[id_bounary_space[i]] = 1/4
    elif cur_nodes[1] == y[1] and cur_nodes[2] == z[0]:
        b0[id_bounary_space[i]] = 1/4
    elif cur_nodes[1] == y[0] and cur_nodes[2] == z[1]:
        b0[id_bounary_space[i]] = 1/4
    elif cur_nodes[1] == y[1] and cur_nodes[2] == z[1]:
        b0[id_bounary_space[i]] = 1/4


def compute_modify_b(b, b0):
    modify_a = -np.sum(b*b0)/np.sum(b0*b0)
    b = b + modify_a*b0
    return b

b = compute_modify_b(b, b0)

A = sc.sparse.coo_matrix.dot(sc.sparse.diags(b0), A)
b = b*b0

A_plus = sc.sparse.lil_matrix((Num_all+1, Num_all+1))
for i in range(Num_all):
    A_plus[i, 0:Num_all] = A[i, 0:Num_all]
A_plus[0:Num_all, Num_all] = h_x*h_y*h_z*b0
A_plus[Num_all, 0:Num_all] = h_x*h_y*h_z*b0.T

b_plus = np.hstack((b, [0]))



from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator, lsqr
# cond_A = np.linalg.cond(A.toarray())
# print(cond_A)

u_h = cg(A_plus, b_plus)[0][0:Num_all]
# u_h = cg(A, b)[0]
# u_h = spsolve(A, b)
u_r = real_u(nodes)

error_inf = np.abs((u_h - u_r)).max()
print(error_inf)

with open('error_inf.txt','w') as f:
   f.write(str(error_inf))

# x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
# fig = plt.figure()
# from mpl_toolkits.mplot3d import Axes3D
# ax = fig.add_subplot(111, projection='3d')
# ww = ax.scatter(x, y, z, c=abs(u_h-u_r), cmap='Blues', marker='o')
# # ww = ax.scatter(x, y, z, c=u_r, cmap='Blues', marker='o')
# fig.colorbar(ww)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.savefig('123.png', bbox_inches='tight')
# # plt.show()








