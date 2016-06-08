# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import approx_fprime

__author__ = 'Sergei'


def find_MEP(fun, x0, tol=1e-6, max_iter=10000):
    force_max = 1.
    itr = 0
    k_sp = 1.
    path = x0
    (M, N) = path.shape
    alpha = .01
    eps = np.sqrt(np.finfo(float).eps)
    while (force_max > tol) and (itr < max_iter):
        temp_1 = path[:, 1:-1] - path[:, :-2]
        norm_temp_1 = np.linalg.norm(temp_1, axis=0)
        temp_1 /= norm_temp_1
        temp_2 = path[:, 2:] - path[:, 1:-1]
        norm_temp_2 = np.linalg.norm(temp_2, axis=0)
        temp_2 /= norm_temp_2
        tau = temp_1 + temp_2
        tau /= np.linalg.norm(tau, axis=0)
        gradient = np.array([-approx_fprime(path[:, j], fun, eps) for j in range(1, N-1)]).transpose()
        grad_trans = gradient - np.array([np.dot(gradient[:, j], tau[:, j]) * tau[:, j] for j in range(N-2)]).transpose()
        dist = k_sp * (norm_temp_2 - norm_temp_1)
        grad_spring = dist * tau
        grad_opt = grad_spring + grad_trans
        force_max = max(np.linalg.norm(grad_opt, axis=0))
        path[:, 1:-1] += alpha * grad_opt
        itr += 1
    if force_max < tol:
        print "MEP was successfully found in %i iterations" % itr
    return path
