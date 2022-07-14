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


# 16
# 32
# 64
# 128
# 256


x = [0, 4]
Num_x = 8
h_x = (x[1]-x[0]) / Num_x

nodes = np.linspace(x[0], x[1], Num_x + 1)
Num_all = nodes.shape[0]


id_all = list(range(Num_all))
id_bounary_sapce = list(np.where(np.logical_or(nodes == x[0], nodes == x[1]))[0])
id_bounary_sapce_intial = list(np.where(nodes == x[0])[0])
id_bounary_sapce_end = list(np.where(nodes == x[1])[0])
id_external = id_bounary_sapce
id_internal = list(set(id_all).difference(id_external))



A = sc.sparse.lil_matrix((Num_all, Num_all))
for i in range(len(id_internal)):
    A[id_internal[i], id_internal[i]] = 2 / (h_x * h_x)
    A[id_internal[i], id_internal[i] - 1] = -1 / (h_x * h_x)
    A[id_internal[i], id_internal[i] + 1] = -1 / (h_x * h_x)



A[id_bounary_sapce_intial[0], id_bounary_sapce_intial[0]] = 2 / (h_x * h_x)
A[id_bounary_sapce_intial[0], id_bounary_sapce_intial[0] + 1] = -2 / (h_x * h_x)

A[id_bounary_sapce_end[0], id_bounary_sapce_end[0]] = 2 / (h_x * h_x)
A[id_bounary_sapce_end[0], id_bounary_sapce_end[0] - 1] = -2 / (h_x * h_x)


def f(x):
    return (np.pi)**2 * np.sin(np.pi*x) #(np.pi)**2 * np.cos(np.pi*x)  #

def real_u(x):
    return np.sin(np.pi*x)  #np.cos(np.pi*x) #

def N_bounary(nodes):
    out = np.zeros((nodes.shape[0]))
    for i in range(nodes.shape[0]):
        cur_x = nodes[i]
        temp = nodes[i]
        if temp == x[0]:
            out[i] = (f(cur_x) - 2 * (np.pi * np.cos(np.pi * cur_x)) / h_x)  #(f(cur_x) - 2*(np.pi*np.sin(np.pi*cur_x))/h_x)  #
        elif temp == x[1]:
            out[i] = (f(cur_x) + 2 * (np.pi * np.cos(np.pi * cur_x)) / h_x)  #(f(cur_x) + 2*(np.pi*np.sin(np.pi*cur_x))/h_x)  #
    return out


b = f(nodes)

b0 = np.ones(Num_all)
b0[0], b0[-1] = 0.5, 0.5
# c = np.dot(b, b0)*h_x - (np.pi*np.cos(np.pi*0)) + (np.pi*np.cos(np.pi*1))
# a = -c/(h_x*(Num_all-3/2))

b[id_bounary_sapce_end] = N_bounary(nodes[id_bounary_sapce_end])
b[id_bounary_sapce_intial] = N_bounary(nodes[id_bounary_sapce_intial])
# b = b + a*b0

A[id_bounary_sapce_intial, :] = A[id_bounary_sapce_intial, :]/2
A[id_bounary_sapce_end, :] = A[id_bounary_sapce_end, :]/2
b[id_bounary_sapce_intial] = b[id_bounary_sapce_intial]/2
b[id_bounary_sapce_end] = b[id_bounary_sapce_end]/2

# ww = id_bounary_sapce_intial[0]
# A[ww, :] = 0
# A[ww, ww] = 1
# b[ww] = 0

A_plus = sc.sparse.lil_matrix((Num_all+1, Num_all+1))
A_plus[0:Num_all, 0:Num_all] = A
A_plus[0:Num_all, Num_all] = h_x*b0
A_plus[Num_all, 0:Num_all] = h_x*b0.T


b_plus = np.hstack((b, [0]))


from scipy.sparse.linalg import spsolve, cg, lsqr, bicg, bicgstab, cgs, spilu, LinearOperator, gmres
cond_A = np.linalg.cond(A.toarray())
print(cond_A)

u_h = cg(A_plus, b_plus)[0][0:Num_all]
# u_h = cg(A, b)[0][0:Num_all]
u_r = real_u(nodes)

error_inf = np.abs((u_h - u_r)).max()
print(error_inf)


plt.plot(nodes, u_h, 'r')
plt.plot(nodes, u_r, 'b')
plt.show()

ww = 1
