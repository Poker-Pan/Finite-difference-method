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


# 32  0.01920299621585264       0.053029287545514725
# 64  0.004772284520468739      0.003218964440079519
# 128 0.0011928559584344356     0.0008035776793720029
# 256 0.00020082180970515395    0.00020082180970382169
# 512 5.0200915919784705e-05    5.0200915919784705e-05

x = [0, 4]
Num_x = 8
h_x = (x[1]-x[0]) / Num_x

nodes_all = np.linspace(x[0], x[1], Num_x + 1)
nodes = nodes_all[0:-1]
Num_all = nodes.shape[0]


id_all = list(range(Num_all))
id_bounary_sapce = list(np.where(np.logical_or(nodes == x[0], nodes == x[1]-h_x))[0])
id_bounary_sapce_intial = list(np.where(nodes == x[0])[0])
id_bounary_sapce_end = list(np.where(nodes == x[1]-h_x)[0])
id_external = id_bounary_sapce
id_internal = list(set(id_all).difference(id_external))



A = sc.sparse.lil_matrix((Num_all, Num_all))
for i in range(len(id_internal)):
    A[id_internal[i], id_internal[i]] = 2 / (h_x * h_x)
    A[id_internal[i], id_internal[i] - 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] + 1] = -1 / (h_x * h_x)


# for i in range(len(id_external)):
#     A[id_external[i], id_external[i]] = 1


A[id_bounary_sapce_intial[0], id_bounary_sapce_intial[0]] = 2 / (h_x * h_x)
A[id_bounary_sapce_intial[0], id_bounary_sapce_intial[0] + 1] = -1 / (h_x * h_x)
A[id_bounary_sapce_intial[0], id_bounary_sapce_end[0]] = -1 / (h_x * h_x)

A[id_bounary_sapce_end[0], id_bounary_sapce_end[0]] = 2 / (h_x * h_x)
A[id_bounary_sapce_end[0], id_bounary_sapce_end[0] - 1] = -1 / (h_x * h_x)
A[id_bounary_sapce_end[0], id_bounary_sapce_intial[0]] = -1 / (h_x * h_x)


def f(x):
    return (np.pi)**2 * np.sin(np.pi*x) #(np.pi)**2 * np.cos(np.pi*x)  #

def real_u(x):
    return np.sin(np.pi*x)  #np.cos(np.pi*x) #



b = f(nodes)

b0 = np.ones(Num_all)

A_plus = sc.sparse.lil_matrix((Num_all+1, Num_all+1))
A_plus[0:Num_all, 0:Num_all] = A
A_plus[0:Num_all, Num_all] = h_x*b0
A_plus[Num_all, 0:Num_all] = h_x*b0.T

# from scipy.integrate import quad
# integrate_u = quad(real_u, x[0], x[1])[0]
b_plus = np.hstack((b, [0]))


from scipy.sparse.linalg import spsolve, cg, lsqr, bicg, bicgstab, cgs
cond_A = np.linalg.cond(A.toarray())
print(cond_A)

u_h = cg(A_plus, b_plus)[0][0:Num_all]

# u_h = cg(A, b)[0][0:Num_all]
u_r = real_u(nodes)

u_r = np.hstack((u_r, [u_r[0]]))
u_h = np.hstack((u_h, [u_h[0]]))


error_inf = np.abs((u_h - u_r)).max()
print(error_inf)

plt.plot(nodes_all, abs((u_h - u_r)))
# plt.plot(nodes_all, u_h)
# plt.plot(nodes_all, u_r)
plt.show()

ww = 1
