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



# 16    0.053029287545516945    0.05569929051461786
# 32    0.012950746721882345    0.013534190285430459
# 64    0.0032189644400866246   0.0033597400388698517
# 128   0.0008035776793982041   0.0008384562655352479

x = [0, 4]
y = [0, 4]

num = 16
Num_x = num
Num_y = num

h_x = (x[1]-x[0]) / Num_x
h_y = (y[1]-y[0]) / Num_y

X = np.linspace(x[0], x[1], Num_x + 1)
Y = np.linspace(y[0], y[1], Num_y + 1)    # time

X = X[1:]

mesh = np.meshgrid(X, Y)
nodes = np.array(list(zip(*(dim.flat for dim in mesh))))
nodes = nodes[:, [1, 0]]
Num_all = nodes.shape[0]


id_all = list(range(Num_all))
id_bounary_corner = list([0, (Num_x-1), Num_all-1-(Num_x-1), Num_all-1])
id_bounary_time_initial = list(set(list(np.where(nodes[:, 0] == x[0])[0])).difference(id_bounary_corner))
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


def f(x, y):
    return 2*((np.pi)**2) * np.cos(np.pi*x)*np.cos(np.pi*y)   #2*((np.pi)**2) * np.sin(np.pi*x)*np.sin(np.pi*y)  #


def real_u(nodes):
    x = nodes[:, 0]
    y = nodes[:, 1]
    return np.cos(np.pi*x)*np.cos(np.pi*y)   #np.sin(np.pi*x)*np.sin(np.pi*y)  #



b = f(nodes[:, 0], nodes[:, 1])

b0 = np.ones(Num_all)
b0[id_bounary_time_initial] = 0.5
b0[id_bounary_time_end] = 0.5
b0[id_bounary_corner] = 0.5

b[id_bounary_corner] = real_u(nodes[id_bounary_corner, :])
b[id_bounary_time_initial] = real_u(nodes[id_bounary_time_initial, :])
b[id_bounary_time_end] = real_u(nodes[id_bounary_time_end, :])


A_plus = sc.sparse.lil_matrix((Num_all+1, Num_all+1))
A_plus[0:Num_all, 0:Num_all] = A
A_plus[0:Num_all, Num_all] = h_x*h_y*b0
A_plus[Num_all, 0:Num_all] = h_x*h_y*b0.T

# from scipy.integrate import dblquad
# integrate_u = dblquad(lambda x, y: np.sin(np.pi*x)*np.sin(np.pi*y), x[0], x[1], y[0], y[1])[0]
b_plus = np.hstack((b, [0]))


from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
cond_A = np.linalg.cond(A.toarray())
print(cond_A)

# M = spilu(A_plus, drop_tol=1e-9)
# M = LinearOperator(M.shape, matvec=M.solve)
# u_h = cg(A_plus, b_plus, M=M)[0][0:Num_all]

# u_h = cg(A_plus, b_plus)[0][0:Num_all]
u_h = spsolve(A, b)
u_r = real_u(nodes)

error_inf = np.abs((u_h - u_r)).max()
print(error_inf)


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(nodes[:, 0], nodes[:, 1], u_h, c='r')
ax.scatter(nodes[:, 0], nodes[:, 1], u_r, c='b')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(x)
plt.ylim(y)
plt.show()
ww = 1
