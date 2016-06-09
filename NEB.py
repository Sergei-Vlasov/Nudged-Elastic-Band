# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import approx_fprime

__author__ = 'Sergei'


def find_MEP(fun, x0, tol=1e-6, max_iter=10000, improved_tangent=True):
    force_max = 1.
    itr = 0
    k_sp = 1.
    path = x0
    (M, N) = path.shape
    alpha = .01
    eps = np.sqrt(np.finfo(float).eps)
    while (force_max > tol) and (itr < max_iter):
        temp_minus = path[:, 1:-1] - path[:, :-2]
        norm_temp_minus = np.linalg.norm(temp_minus, axis=0)
        temp_plus = path[:, 2:] - path[:, 1:-1]
        norm_temp_plus = np.linalg.norm(temp_plus, axis=0)
        if improved_tangent:
            energy = fun(path)
            V_max = np.array([max(abs(energy[i + 1] - energy[i]), abs(energy[i - 1] - energy[i]))
                              for i in range(1, N - 1)])
            V_min = np.array([min(abs(energy[i + 1] - energy[i]), abs(energy[i - 1] - energy[i]))
                              for i in range(1, N - 1)])
            i_minus = np.array([i for i in range(1, N - 1) if energy[i + 1] < energy[i] < energy[i - 1]]) - 1
            i_plus = np.array([i for i in range(1, N - 1) if energy[i - 1] < energy[i] < energy[i + 1]]) - 1
            i_mix = np.array([i for i in range(N - 2) if i not in np.hstack((i_minus, i_plus))])
            i_mix_minus = np.array([i for i in i_mix if energy[i] > energy[i + 2]])
            i_mix_plus = np.array([i for i in i_mix if energy[i + 2] > energy[i]])
            tau = np.zeros((M, N - 2))
            if len(i_minus) > 0:
                tau[:, i_minus] = temp_minus[:, i_minus]
            if len(i_plus) > 0:
                tau[:, i_plus] = temp_plus[:, i_plus]
            if len(i_mix_minus) > 0:
                tau[:, i_mix_minus] = temp_plus[:, i_mix_minus] * V_min[i_mix_minus] + \
                                      temp_minus[:, i_mix_minus] * V_max[i_mix_minus]
            if len(i_mix_plus) > 0:
                tau[:, i_mix_plus] = temp_plus[:, i_mix_plus] * V_max[i_mix_plus] + \
                                     temp_minus[:, i_mix_plus] * V_min[i_mix_plus]
            tau /= np.linalg.norm(tau, axis=0)
        else:
            temp_minus /= norm_temp_minus
            temp_plus /= norm_temp_plus
            tau = temp_minus + temp_plus
            tau /= np.linalg.norm(tau, axis=0)
        gradient = np.array([-approx_fprime(path[:, j], fun, eps) for j in range(1, N-1)]).transpose()
        grad_trans = gradient - np.array([np.dot(gradient[:, j], tau[:, j]) * tau[:, j] for j in range(N-2)]).transpose()
        dist = k_sp * (norm_temp_plus - norm_temp_minus)
        grad_spring = dist * tau
        grad_opt = grad_spring + grad_trans
        force_max = max(np.linalg.norm(grad_opt, axis=0))
        path[:, 1:-1] += alpha * grad_opt
        itr += 1
    if force_max < tol:
        print "MEP was successfully found in %i iterations" % itr
    return path
