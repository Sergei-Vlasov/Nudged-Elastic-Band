# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from NEB import find_MEP
from scipy.optimize import minimize
import random
__author__ = 'Sergei'


def Q(x, d):
    alpha = 1.942
    x0 = .742
    return .5 * d * (1.5 * np.exp(-2 * alpha * (x - x0)) - np.exp(-alpha * (x - x0)))


def J(x, d):
    alpha = 1.942
    x0 = .742
    return .25 * d * (np.exp(-2 * alpha * (x - x0)) - 6 * np.exp(-alpha * (x - x0)))


def V_LEPS(r_AB, r_AC):
    a = .05 + 1
    b = .8 + 1
    c = .05 + 1
    d_AB = 4.746
    d_BC = 4.746
    d_AC = 3.445
    r_BC = r_AC - r_AB
    return Q(r_AB, d_AB) / a + Q(r_BC, d_BC) / b +  Q(r_AC, d_AC) / c - \
           np.sqrt((J(r_AB, d_AB) / a) ** 2 + (J(r_BC, d_BC) / b) ** 2 + (J(r_AC, d_AC) / c) ** 2 -
                    J(r_AB, d_AB) * J(r_BC, d_BC) / a / b - J(r_AB, d_AB) * J(r_AC, d_AC) / a / c -
                    J(r_AC, d_AC) * J(r_BC, d_BC) / c / b)


def f(x):
    k_c = .2025
    c = 1.154
    r_AC = 3.742
    return V_LEPS(x[0], r_AC) + 2 * k_c * (x[0] - .5 * r_AC + x[1] / c) ** 2

delta = 0.1
x_initial = minimize(f, [0.75, 1.0]).x
x_final = minimize(f, [3.0, -1.0]).x
N = 12
x0 = np.array([x_initial + (x_final - x_initial) * i / (N-1) for i in range(N)]).transpose()
for i in range(2):
    for j in range(1, N-1):
        x0[i, j] += random.uniform(0, delta)
n = 100
x = np.linspace(0.5, 3.2, n)
y = np.linspace(-3, 3, n)
X,Y = np.meshgrid(x, y)
plt.figure(1)
plt.contourf(X, Y, f([X, Y]), 15, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(X, Y,f([X, Y]), 15, colors='black', linewidth=.5)
plt.scatter(x0[0,:], x0[1,:])
plt.show()

min_en_path = find_MEP(f, x0)

plt.contourf(X, Y, f([X, Y]), 15, alpha=.75, cmap=plt.cm.hot)
C = plt.contour(X, Y,f([X, Y]), 15, colors='black', linewidth=.5)
plt.plot(min_en_path[0,:], min_en_path[1,:], 'o-')
plt.show()
