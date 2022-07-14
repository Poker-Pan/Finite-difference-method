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



# 8    0.28697052847603244     0.23370055013617064
# 16   0.062303466524194606    0.05302928754551606
# 32   0.015053074700831948    0.01295074672188501
# 64   0.0037315372256940815   0.0032189644400959505
# 128  0.0009319278504227313   0.0008035776794390603

x = [0, 4]
y = [0, 4]

num = 8
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
id_bounary_time_initial = list(np.where(nodes[:, 0] == x[0])[0])
id_bounary_time_end = list(np.where(nodes[:, 0] == x[1])[0])
id_bounary_time = list(set(id_bounary_time_initial).union(id_bounary_time_end))
id_bounary_space = list(set(list(np.where(np.logical_or(nodes[:, 1] == y[0], nodes[:, 1] == y[1]))[0])).difference(id_bounary_time))
id_external = list(set(id_bounary_time).union(id_bounary_space))
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


def f(x, y):
    return 2*((np.pi)**2) * np.cos(np.pi*x)*np.cos(np.pi*y)   #2*((np.pi)**2) * np.sin(np.pi*x)*np.sin(np.pi*y)  #

def real_u(nodes):
    x = nodes[:, 0]
    y = nodes[:, 1]
    return np.cos(np.pi*x)*np.cos(np.pi*y)   #np.sin(np.pi*x)*np.sin(np.pi*y)  #

b = f(nodes[:, 0], nodes[:, 1])
b[id_external] = real_u(nodes[id_external, :])


from scipy.sparse.linalg import spsolve, cg, lsqr
cond_A = np.linalg.cond(A.toarray())
print(cond_A)

u_h = spsolve(A, b)
u_r = real_u(nodes)

error_inf = np.abs((u_h - u_r)).max()
print(error_inf)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(nodes[:, 0], nodes[:, 1], u_h, c='r')
# ax.scatter(nodes[:, 0], nodes[:, 1], u_r, c='b')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

ww = 1
