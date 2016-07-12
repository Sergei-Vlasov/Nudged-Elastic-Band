# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

__all__ = ['find_MEP']

import numpy as np
from scipy.optimize import approx_fprime


__author__ = 'Sergei'


def calculate_tangent(fun, x, improved_tangent):
    (M, N) = x.shape
    temp_minus = x[:, 1:-1] - x[:, :-2]
    norm_temp_minus = np.linalg.norm(temp_minus, axis=0)
    temp_plus = x[:, 2:] - x[:, 1:-1]
    norm_temp_plus = np.linalg.norm(temp_plus, axis=0)
    if improved_tangent:
        energy = fun(x)
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
    return tau


def update_SD(x, g, tau, alpha):
    (M, N) = x.shape
    grad_trans = g - np.array([np.dot(g[:, j], tau[:, j]) * tau[:, j] for j in range(N - 2)]).transpose()
    force_max = max(np.linalg.norm(grad_trans, axis=0))
    x[:, 1:-1] -= alpha * grad_trans
    return x, force_max


def repartition_tagentaware_proj(x, tau):
    dx = x[:, 1:] - x[:, :-1]
    beta = -0.5 * ((dx[:, 1:] - dx[:, :-1]) * (dx[:, 1:] + dx[:, :-1])).sum(axis=0)
    plus = (tau * dx[:, 1:]).sum(axis=1)
    minus = (tau * dx[:, :-1]).sum(axis=1)
    alpha = np.zeros((x.shape[1] - 2, x.shape[1] - 2))
    np.fill_diagonal(alpha[:-1, 1:], minus[1:])
    np.fill_diagonal(alpha[1:, :-1], plus[1:])
    np.fill_diagonal(alpha, -plus[1:] - minus[:-1])
    a = np.linalg.solve(alpha, beta)
    x[:, 1:-1] += tau * a
    return x


def repartition_spring(x, tau, k):
    temp_minus = x[:, 1:-1] - x[:, :-2]
    norm_temp_minus = np.linalg.norm(temp_minus, axis=0)
    temp_plus = x[:, 2:] - x[:, 1:-1]
    norm_temp_plus = np.linalg.norm(temp_plus, axis=0)
    dist = k * (norm_temp_plus - norm_temp_minus)
    x[:, 1:-1] += dist * tau
    return x


def repartition_repar(x):
    (M, N) = x.shape
    L = np.concatenate(([0], np.cumsum(np.linalg.norm(x[:, 1:] - x[:, :-1], axis=0))))
    L /= L[-1]
    s = np.linspace(0, 1, N)
    z = np.array([np.interp(s, L, x[j, :]) for j in range(M)])
    return z


def find_MEP(fun, x0, args=(), jac=None, tol=1e-6, max_iter=10000,
             improved_tangent=True, method=None, repartition='spring'):

    x0 = np.asarray(x0)
    eps = np.sqrt(np.finfo(float).eps)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    def fun_wrapped(x):
        return np.atleast_1d(fun(x, *args))

    if jac is None:
        jac = approx_fprime
        kwargs = (fun_wrapped, eps)
    else:
        kwargs = args

    if method is None:
        method = 'SD'

    meth = method.lower()
    repart = repartition.lower()

    force_max = 1.
    itr = 0
    k_sp = .01
    path = x0
    (M, N) = path.shape
    alpha = .01
    while (force_max > tol) and (itr < max_iter):
        tau = calculate_tangent(fun_wrapped, path, improved_tangent)
        gradient = np.array([jac(path[:, j], *kwargs) for j in range(1, N-1)]).transpose()

        if meth == 'sd':
            path, force_max = update_SD(path, gradient, tau, alpha)

        if repart == 'spring':
            path = repartition_spring(path, tau, k_sp)
        elif repart == 'repar':
            path = repartition_repar(path)
        elif repart == 'repar_tangent':
            path = repartition_tagentaware_proj(path, tau)
        itr += 1
    if force_max < tol:
        print("MEP was successfully found in %i iterations" % itr)
    else:
        print("Max. force projection: %f" % force_max)
    return path
