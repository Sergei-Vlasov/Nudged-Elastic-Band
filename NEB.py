# -*- coding: utf-8 -*-
__author__ = 'Sergei'

import numpy as np
from scipy.signal import argrelextrema
from numpy import pi
from numpy import cos
from numpy import sin
from numpy import sqrt
import json
import matplotlib.pyplot as plt

with open('params1.json') as json_data:
    params = json.load(json_data)
    json_data.close()

n = params['system']['n'] # Number of moments

factor = 1.e11
M = params['system']['M'] # Magnetisation
K = params['system']['K']*2.0 # Anisotropy from my calculations
# KV05 = 1.482e-12*factor
mu0 = 1.0  # Magnetic constant
R = params['system']['R'] # Radius of hexagon
J = params['system']['J']
H = params['system']['H'] # The value of an external magnetic field (1 µT = 10 Gauss)
V = params['system']['V']
KV05 = K*V*0.5*factor
HMV = H * V * M * factor
mu0 = factor

k_spring = params['programm']['k_spring']
use_old_fashion_tangent = False

anisotropy_angles = np.array([0, 5 * pi / 3, 4 * pi / 3, pi, 2 * pi / 3, pi / 3])
field_angle = params['system']['H_angle'] # From left to right is the zero

distances = np.array([
    0          , R          , sqrt(3) * R, 2 * R      , sqrt(3) * R, R,
    R          , 0          , R          , sqrt(3) * R, 2 * R      , sqrt(3) * R,
    sqrt(3) * R, R          , 0          , R          , sqrt(3) * R, 2 * R,
    2 * R      , sqrt(3) * R, R          , 0          , R          , sqrt(3) * R,
    sqrt(3) * R, 2 * R      , sqrt(3) * R, R          , 0          , R,
    R          , sqrt(3) * R, 2 * R      , sqrt(3) * R, R          , 0

])
distances = distances.reshape((6, 6))  # Reshape it to matrix for easier access

distance_unit_vectors = np.array([
    0         , 11 * pi / 6, 10 * pi / 6, 9 * pi / 6, 8 * pi / 6, 7 * pi / 6,
    5 * pi / 6, 0          , 9 * pi / 6 , 8 * pi / 6, 7 * pi / 6, pi,
    4 * pi / 6, 3 * pi / 6 , 0          , 7 * pi / 6, pi, 5 * pi / 6,
    3 * pi / 6, 2 * pi / 6 , pi / 6     , 0         , 5 * pi / 6, 4 * pi / 6,
    2 * pi / 6, pi / 6     , 0          , 5 * pi / 6, 0, 3 * pi / 6,
    pi / 6    ,           0, 5 * pi / 6 , 4 * pi / 6, 3 * pi / 6, 0

])
distance_unit_vectors = distance_unit_vectors.reshape((6, 6))  # Reshape it to matrix for easier access


def anisotropy_energy(system_angles):
    return KV05 * (cos(system_angles - anisotropy_angles) ** 2).sum()


def zeeman_energy(system_angles):
    return HMV * cos(system_angles - field_angle).sum()

def dipole_dipole_energy(system_angles):
    result = 0.0
    for i in range(0, n):
        x = system_angles[i]
        for j in range(0, n):
            if i != j:
                dist_xy = distances[i, j]
                dist_xy_vector = distance_unit_vectors[i, j]
                y = system_angles[j]
                temp_first = 3. * cos(x - dist_xy_vector) * cos(y - dist_xy_vector) - cos(x - y)
                temp_second = temp_first * (M * M * V * V * mu0 / (dist_xy ** 3))  # / 4. / pi)
                result += temp_second
    return result * 0.5

def exchange_energy(system_angles):
    return J*M*M*cos(system_angles[0:n-2]-system_angles[1:n-1]).sum()

def energy(system_angles):
    return -anisotropy_energy(system_angles) - dipole_dipole_energy(system_angles) - zeeman_energy(system_angles) #- exchange_energy(system_angles)


def gradient_dipole_exact(system_angles):
    gradient = np.zeros(n)
    for i in range(0, n):
        dipole_gradient = 0.0
        x = system_angles[i]
        for j in range(0, n):
            if i != j:
                dist_xy = distances[i, j]
                dist_xy_vector = distance_unit_vectors[i, j]
                y = system_angles[j]
                temp_first = sin(x - y) - 3. * sin(x - dist_xy_vector) * cos(y - dist_xy_vector)
                temp_second = temp_first * (M * M * V * V * mu0 / (dist_xy ** 3))  # / 4. / pi)
                dipole_gradient += temp_second
        gradient[i] = dipole_gradient
    return gradient


def gradient_anisotropy_exact(system_angles):
    gradient = np.zeros(n)
    for i in range(0, n):
        gradient[i] = KV05 * sin(2. * (system_angles[i] - anisotropy_angles[i]))
    return gradient


def gradient_zeeman_exact(system_angles):
    gradient = np.zeros(n)
    for i in range(0, n):
        gradient[i] = HMV * sin((system_angles[i] - field_angle))
    return gradient

def gradient_exchange(system_angles):
    gradient = np.zeros(n)
    for i in range(1, n-2):
        gradient[i] = J*M*M*(sin((system_angles[i-1] - system_angles[i])) + sin(system_angles[i+1]-system_angles[i]))
    gradient[0] = J*M*M*sin((system_angles[1] - system_angles[0]))
    gradient[n-1] = J*M*M*sin((system_angles[n-2] - system_angles[n-1]))
    return gradient

def gradient_exact(system_angles):
    return +gradient_zeeman_exact(system_angles) + gradient_anisotropy_exact(system_angles) - gradient_dipole_exact(
        system_angles) #+ gradient_exchange(system_angles)


def hessian(system_angles):
    hes = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                anis = 2 * KV05 * cos(2. * (system_angles[i] - anisotropy_angles[j]))
                zee = HMV * cos(system_angles[i] - field_angle)
                dip = 0.
                x = system_angles[i]
                for di in range(0, n):
                    if di != i:
                        y = system_angles[di]
                        dist_xy = distances[i, di]
                        dist_xy_vector = distance_unit_vectors[i, di]
                        temp_first = -3. * cos(x - dist_xy_vector) * cos(y - dist_xy_vector) + cos(x - y)
                        temp_second = temp_first * (M * M * V * V * mu0 / (dist_xy ** 3))  # / 4. / pi)
                        dip += temp_second
                hes[i, j] = anis - dip + zee
            else:
                zee = 0  # Mixed derivative from Zeeman energy equals to zero
                x = system_angles[i]
                y = system_angles[j]
                dist_xy = distances[i, j]
                dist_xy_vector = distance_unit_vectors[i, j]
                temp_first = 3. * sin(x - dist_xy_vector) * sin(y - dist_xy_vector) - cos(x - y)
                dip = temp_first * (M * M *V*V* mu0 / (dist_xy ** 3) )#/ 4. / pi)
                hes[i, j] = - dip
    return hes


def tangent_unit_vectors_calc(state_path, energy_path):
    tangents = np.zeros((n, m))
    if not use_old_fashion_tangent:

        tangent_plus = state_path[0:n, 2:m] - state_path[0:n, 1:m - 1]
        tangent_minus = state_path[0:n, 1:m - 1] - state_path[0:n, 0:m - 2]

        for i in range(1, m - 1):
            t_index = i - 1
            if energy_path[i + 1] > energy_path[i] > energy_path[i - 1]:
                tangents[:, i] = tangent_plus[:, t_index]
            elif energy_path[i + 1] < energy_path[i] < energy_path[i - 1]:
                tangents[:, i] = tangent_minus[:, t_index]
            else:
                e_max = max(abs(energy_path[i + 1] - energy_path[i]), abs(energy_path[i - 1] - energy_path[i]))
                e_min = min(abs(energy_path[i + 1] - energy_path[i]), abs(energy_path[i - 1] - energy_path[i]))
                if energy_path[i + 1] > energy_path[i - 1]:
                    tangents[:, i] = tangent_plus[:, t_index] * e_max + tangent_minus[:, t_index] * e_min
                else:
                    tangents[:, i] = tangent_plus[:, t_index] * e_min + tangent_minus[:, t_index] * e_max
        tangents_norms = np.linalg.norm(tangents[0:n, 1:m - 1], axis=0)
        return tangents[0:n, 1:m - 1] / tangents_norms[None:, ]
        # return tangents[0:n, 1:m - 1]

    else:
        te = state_path[0:n, 2:m] - state_path[0:n, 0:m - 2]
        tangents_norms = np.linalg.norm(te, axis=0)
        return te / tangents_norms[None:, ]


def spring_forces(state_path, tangent_unit_vectors):
    tangent_plus = state_path[0:n, 2:m] - state_path[0:n, 1:m - 1]
    tangent_minus = state_path[0:n, 1:m - 1] - state_path[0:n, 0:m - 2]
    norm_spr = k_spring * np.linalg.norm(tangent_plus, axis=0) - np.linalg.norm(tangent_minus, axis=0)
    return tangent_unit_vectors * norm_spr[None:, ]


def perpendicular_forces(state_path, tangent_unit_vectors):
    gradients = np.zeros((n, m))
    tangents_gradients_dot_product = np.zeros(m)
    for i in range(1, m - 1):
        gradients[:, i] = gradient_exact(state_path[:, i])
        tangents_gradients_dot_product[i] = np.dot(gradients[:, i], tangent_unit_vectors[:, i - 1])
    gradients = gradients[0:n, 1:m - 1]
    tangents_gradients_dot_product = tangents_gradients_dot_product[1:m - 1]
    temp = tangent_unit_vectors * tangents_gradients_dot_product[None:, ]
    return gradients - temp


def true_forces(state_path, tangent_unit_vectors):
    return -perpendicular_forces(state_path, tangent_unit_vectors) + spring_forces(state_path, tangent_unit_vectors)


def ci_true_forces(max_image, tan_unit_vec):
    return -gradient_exact(max_image) + (2 * np.dot(gradient_exact(max_image), tan_unit_vec) * tan_unit_vec)


def fi_true_forces(min_image, tan_unit_vec):
    return -gradient_exact(min_image) - (2 * np.dot(gradient_exact(min_image), tan_unit_vec) * tan_unit_vec)


def argmin(paths):
    en = np.zeros(m)
    for i in range(0, m - 2):
        en[i] = energy(paths[:, i])
    return np.min(en)


def neb_calc(s_p, dt=0.05, epsilon=1.e-5, use_ci=True, use_fi=True):
    # epsilon = 10e-5
    mass = 1.0
    TOL = np.max(epsilon, m ** -4)
    #dt = 0.05*np.min(0.2, 1/m)
    TOL = np.max(epsilon, m ** -4)
    d = 1.
    maxForce = 1.
    j = 1
    state_path = np.copy(s_p)
    new_state_path = np.copy(s_p)
    energy_path = np.zeros(m)

    while np.abs(maxForce) > TOL:
        if j % 100 == 0:
            print d
            print dt
            print maxForce
            print "step ", j, '\n'

        # if j % 1000 == 0:
        #         plt.plot(range(0, m), energy_path)
        #         plt.draw()
        #         plt.pause(0.0001)

        for i in range(0, m):
            energy_path[i] = energy(state_path[:, i])

        tangent_unit_vectors = tangent_unit_vectors_calc(state_path, energy_path)
        forces = true_forces(state_path, tangent_unit_vectors)
        maximus = argrelextrema(energy_path, np.greater)
        minimums = argrelextrema(energy_path, np.less)


        if j > 300:
            if use_ci:
                for max_en_image_number in [item for sublist in maximus for item in sublist]:
                    forces[:, max_en_image_number - 1 ] = ci_true_forces(state_path[:, max_en_image_number],
                                                                    tangent_unit_vectors[:, max_en_image_number - 1])
            if use_fi:
                for min_en_image_number in [item for sublist in minimums for item in sublist]:
                    forces[:, min_en_image_number - 1 ] = fi_true_forces(state_path[:, min_en_image_number],
                                                                    tangent_unit_vectors[:, min_en_image_number - 1])


        #fun = lambda x: argmin((state_path[0:n, 1:m - 1] + x*forces)%(2*pi))
        #g = spo.minimize_scalar(fun)
        #dt = g.x

        #print dt
        new_state_path[:, 1:m - 1] = state_path[0:n, 1:m - 1] + dt * forces
        new_state_path %= 2 * pi
        new_state_path[:, 0] = s_p[:, 0]
        new_state_path[:, m - 1] = s_p[:, m - 1]

        #d = np.max(np.linalg.norm((new_state_path-state_path), axis=0)/np.abs(dt))
        maxForce = np.max(forces)

        state_path = np.copy(new_state_path)
        j += 1
    return state_path
m = params['programm']['m'] #H = 0
# m = 12 H = 15, 25
# m = 14 H = 20
#m = 8 # H = 35

aa = np.array([0, 5 * pi / 3, 4 * pi / 3, pi, 2 * pi / 3, pi / 3])
#aa = np.array([ 2.53460456, 1.90769476, 1.23900953, 2.53460456, 1.90769476, 1.23900953])

bb = np.array([0 + pi, 5 * pi / 3 + pi, 4 * pi / 3 + pi, pi + pi, 2 * pi / 3 + pi, pi / 3 + pi])%(2 * pi)
#bb = np.array([ 0.6069881, 1.90258312 ,1.23389789 ,0.6069881 , 1.90258312, 1.23389789])

state_0 = aa
state_m = bb
diff = state_m - state_0

state_path = np.zeros((n, m))
state_path[:, 0] = state_0
state_path[:, m - 1] = state_m
for j in range(1, m - 1):
    state_path[:, j] = state_path[:, j - 1] + diff / m
state_path %= (2 * np.pi)  # Remove 2pi phase
#state_path = np.load("{}path{:.2f}+{:.1f}.npy".format(params['programm']['output_path'], params['system']['H_angle'], H))


x = neb_calc(state_path, 0.1, 1.e-8, use_ci=True, use_fi=True)

energy_path = np.zeros(x.shape[1])
for i in range(0, x.shape[1]):
    energy_path[i] = energy(x[:, i])

plt.plot(range(0, x.shape[1]), energy_path)
plt.show()

# m = 8 H = 20,25,30,35
m = params['programm']['m_max']
maximums = argrelextrema(energy_path, np.greater_equal)
print maximums
mep = (x[:, maximums])[:, 0]

if not np.array_equal(mep[:, 0], aa):
    mep = np.concatenate((aa.reshape((n, 1)), mep), axis=1)
if not np.array_equal(mep[:, mep.shape[1]-1], bb):
    mep = np.concatenate((mep, bb.reshape((n, 1))), axis=1)

columns = mep.shape[1]
path = np.zeros((n, m))
for i in range(0, (columns - 1)):
    print 'step ', i, 'begin'
    state_0 = mep[:, i]
    state_m = mep[:, i + 1]
    diff = state_m - state_0

    state_path = np.zeros((n, m))
    state_path[:, 0] = state_0
    state_path[:, m - 1] = state_m
    for j in range(1, m - 1):
        state_path[:, j] = state_path[:, j - 1] + diff / m
    state_path %= (2 * np.pi)  # Remove 2pi phase
    x = neb_calc(state_path, epsilon=1.e-10)

    if i == 0:
        path = np.copy(x)
    else:
        path = np.concatenate((path, x[:, 1:m]), axis=1)
    print 'step ', i, 'end'

energy_path = np.zeros(path.shape[1])
for i in range(0, path.shape[1]):
    energy_path[i] = energy(path[:, i])

plt.plot(range(0, path.shape[1]), energy_path)
plt.show()

# np.save("/Users/Sergei/path{:+f}{:+f}ta-1.npy".format(H, field_angle), path) # only need it with 30 field
# m = 8  H = 30
m = params['programm']['m_min']

minimums = argrelextrema(energy_path, np.less_equal)
print minimums
mep = (path[:, minimums])[:, 0]


columns = mep.shape[1]
path = np.zeros((n, m))
for i in range(0, (columns - 1)):
    print 'step ', i, 'begin'
    state_0 = mep[:, i]
    state_m = mep[:, i + 1]
    diff = state_m - state_0

    state_path = np.zeros((n, m))
    state_path[:, 0] = state_0
    state_path[:, m - 1] = state_m
    for j in range(1, m - 1):
        state_path[:, j] = state_path[:, j - 1] + diff / m
    state_path %= (2 * np.pi)  # Remove 2pi phase
    x = neb_calc(state_path, epsilon=1.e-10)

    if i == 0:
        path = np.copy(x)
    else:
        path = np.concatenate((path, x[:, 1:m]), axis=1)
    print 'step ', i, 'end'

if not np.array_equal(path[:, 0], aa):
    path = np.concatenate((aa.reshape((n, 1)), path), axis=1)
if not np.array_equal(path[:, path.shape[1]-1], bb):
    path = np.concatenate((path, bb.reshape((n, 1))), axis=1)



np.save("{}path{:.2f}+{:.1f}.npy".format(params['programm']['output_path'], field_angle, H), path)

#path = np.concatenate((aa.reshape((n, 1)), path), axis=1)
#path = np.concatenate((path, bb.reshape((n, 1))), axis=1)

energy_path = np.zeros(path.shape[1])
for i in range(0, path.shape[1]):
    energy_path[i] = energy(path[:, i])/factor/1.60217657e-12

np.save("{}energy_path{:.2f}+{:.1f}.npy".format(params['programm']['output_path'], field_angle, H), energy_path)

plt.figure(num=None, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
plt.ylim([np.min(energy_path) - .1, np.max(energy_path) + .1])
plt.ylabel('eV')
plt.xlim([0-3, path.shape[1]+2])
plt.plot(range(0, path.shape[1]), energy_path, linewidth=2)
#plt.scatter(minimums , minimum_energy_path[minimums], marker='o',c = 'r',linewidth=3)
plt.show()
