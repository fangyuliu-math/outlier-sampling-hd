#!/usr/bin/env python

"""
More' and Wild Test Set
Lindon Roberts, 2017

This is a collection nonlinear (and nonconvex) least-squares problems
with low dimension (2 <= n <= 12), designed or testing derivative-free
solvers. As such, we do not provide any derivatives of the objectives.

This code can either provide the vector of residuals or of full least-squares
objective. That is, for a problem
     min_x   f(x) = r_1(x)^2 + ... + r_m(x)^2
this code can give you either the vector function
    x -> [r_1(x) ... r_m(x)]
or the scalar function
    x -> f(x)

More details of the problems, including an estimate of f_min, are given in
the associated file MoreWild_info.csv. For more information, see the original
paper [1].

** Usage **
Inputs:
- problem number from 1, ..., 53

Call sequence:
- Option 1: get scalar function f(x)
    from more_wild import *
    objfun, x0, n, m = get_problem_as_scalar_objective(probnum)

- Option 2: get vector function of residuals
    from more_wild import *
    objfun, x0, n, m = get_problem_as_residual_vector(probnum)

Outputs:
- objfun is a Python function which returns either f(x) or [r_1(x) ... r_m(x)]
  which is called as "f = objfun(x)" or "rvec = objfun(x)"
- x0 is a NumPy vector with a initial starting point for the solver
- n is the dimension of the problem, i.e. len(x0)
- m is the number of residuals in the objective. If returning the vector
  function of residuals, m = len(objfun(x)).


Translated from the original Matlab code (http://www.mcs.anl.gov/~more/dfo/).

References:
[1]  J. J. More' and S. M. Wild, Benchmarking Derivative-Free Optimization Algorithms,
     SIAM J. Optim., 20 (2009), pp. 172-191.
"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

from math import sqrt, pi, exp, log, sin, cos, atan
import logging
import numpy as np
import sys  # for max floating point (gracefully handle overflow error)

__all__ = ['get_problem_as_scalar_objective', 'get_problem_as_residual_vector']

# A list of the input settings
setting_list = np.array([[1, 9, 45, 0],
                         [1, 9, 45, 1],
                         [2, 7, 35, 0],
                         [2, 7, 35, 1],
                         [3, 7, 35, 0],
                         [3, 7, 35, 1],
                         [4, 2, 2, 0],
                         [4, 2, 2, 1],
                         [5, 3, 3, 0],
                         [5, 3, 3, 1],
                         [6, 4, 4, 0],
                         [6, 4, 4, 1],
                         [7, 2, 2, 0],
                         [7, 2, 2, 1],
                         [8, 3, 15, 0],
                         [8, 3, 15, 1],
                         [9, 4, 11, 0],
                         [10, 3, 16, 0],
                         [11, 6, 31, 0],
                         [11, 6, 31, 1],
                         [11, 9, 31, 0],
                         [11, 9, 31, 1],
                         [11, 12, 31, 0],
                         [11, 12, 31, 1],
                         [12, 3, 10, 0],
                         [13, 2, 10, 0],
                         [14, 4, 20, 0],
                         [14, 4, 20, 1],
                         [15, 6, 6, 0],
                         [15, 7, 7, 0],
                         [15, 8, 8, 0],
                         [15, 9, 9, 0],
                         [15, 10, 10, 0],
                         [15, 11, 11, 0],
                         [16, 10, 10, 0],
                         [17, 5, 33, 0],
                         [18, 11, 65, 0],
                         [18, 11, 65, 1],
                         [19, 8, 8, 0],
                         [19, 10, 12, 0],
                         [19, 11, 14, 0],
                         [19, 12, 16, 0],
                         [20, 5, 5, 0],
                         [20, 6, 6, 0],
                         [20, 8, 8, 0],
                         [21, 5, 5, 0],
                         [21, 5, 5, 1],
                         [21, 8, 8, 0],
                         [21, 10, 10, 0],
                         [21, 12, 12, 0],
                         [21, 12, 12, 1],
                         [22, 8, 8, 0],
                         [22, 8, 8, 1]], dtype=int)


def get_problem_as_scalar_objective(probnum, noise_type='smooth', noise_level=1e-2):
    setting_vector = setting_list[probnum - 1, :]
    resid_objfun, x0, nprob, n, m = get_objfun_and_x0_from_settings(setting_vector, noise_type=noise_type,
                                                                    noise_level=noise_level)
    sum_of_squares = lambda x: np.dot(x, x)
    full_objfun = lambda x: sum_of_squares(resid_objfun(x))
    return full_objfun, x0, n, m


def get_problem_as_residual_vector(probnum, noise_type='smooth', noise_level=1e-2):
    setting_vector = setting_list[probnum - 1, :]
    resid_objfun, x0, nprob, n, m = get_objfun_and_x0_from_settings(setting_vector, noise_type=noise_type,
                                                                    noise_level=noise_level)
    return resid_objfun, x0, n, m


def get_objfun_and_x0_from_settings(setting_vector, noise_type='smooth', noise_level=1e-2):
    assert len(setting_vector) == 4, "setting_vector must be of length 4"
    nprob = setting_vector[0]
    n = setting_vector[1]
    m = setting_vector[2]
    factor = 10 ** setting_vector[3]

    x0 = dfoxs(n, nprob, init_factor=factor)
    objfun = lambda x: add_noise(x, dfovec(x, m, nprob), probtype=noise_type, sigma=noise_level)
    return objfun, x0, nprob, n, m


def add_noise(x, rvec, probtype='smooth', sigma=1e-2):
    m = len(rvec)
    if probtype == 'smooth':
        return rvec
    elif probtype == 'multiplicative_deterministic':
        return sqrt(1 + sigma * deterministic_noise(x)) * rvec
    elif probtype == 'multiplicative_uniform':
        return (1 + sigma * np.random.uniform(-1.0, 1.0, (m,))) * rvec
    elif probtype == 'multiplicative_gaussian':
        return (1 + sigma * np.random.normal(0.0, 1.0, (m,))) * rvec
    elif probtype == 'additive_chi_square':
        return np.sqrt(rvec ** 2 + sigma ** 2 * np.random.normal(0.0, 1.0, (m,)) ** 2)
    elif probtype == 'additive_gaussian':
        return rvec + sigma * np.random.normal(0.0, 1.0, (m,))
    else:
        raise ValueError("Unknown noise type %s" % probtype)


def deterministic_noise(x):
    # https://doi.org/10.1137/080724083, eqns (4.3) and (4.4)
    x_norm1 = np.sum(np.abs(x))
    x_norm2 = np.dot(x, x)
    x_normInf = np.max(np.abs(x))
    pi0 = 0.9 * sin(100.0 * x_norm1) * cos(100.0 * x_normInf) + 0.1 * cos(x_norm2)
    return pi0 * (4.0 * pi0 ** 3 - 3.0)


def dfovec(x, m, prob_number):
    # Evaluate objective at point x
    n = np.size(x)
    assert (np.shape(x) == (n,)), "x must be of shape (n,)"

    problem_number_map = {}
    problem_number_map[1] = objective_linear_full_rank
    problem_number_map[2] = objective_linear_rank1
    problem_number_map[3] = objective_linear_rank1_with_zeros
    problem_number_map[4] = objective_rosenbrock
    problem_number_map[5] = objective_helical_valley
    problem_number_map[6] = objective_powell_singular
    problem_number_map[7] = objective_freudenstein_roth
    problem_number_map[8] = objective_bard
    problem_number_map[9] = objective_kowalik_osborne
    problem_number_map[10] = objective_meyer
    problem_number_map[11] = objective_watson
    problem_number_map[12] = objective_box_3d
    problem_number_map[13] = objective_jennrich_sampson
    problem_number_map[14] = objective_brown_dennis
    problem_number_map[15] = objective_chebyquad
    problem_number_map[16] = objective_brown_almost_linear
    problem_number_map[17] = objective_osborne1
    problem_number_map[18] = objective_osborne2
    problem_number_map[19] = objective_bdqrtic
    problem_number_map[20] = objective_cube
    problem_number_map[21] = objective_mancino
    problem_number_map[22] = objective_heart8ls

    if prob_number not in problem_number_map:
        raise ValueError("Unknown problem number (must be in 1, ..., 22)")

    # Calculate objective value
    fvec = problem_number_map[prob_number](x, n, m)
    return fvec


# Initial point x0
def dfoxs(n, prob_number, init_factor=1.0):
    # Get initial point

    problem_number_map = {}
    problem_number_map[1] = initial_value_linear_full_rank
    problem_number_map[2] = initial_value_linear_rank1
    problem_number_map[3] = initial_value_linear_rank1_with_zeros
    problem_number_map[4] = initial_value_rosenbrock
    problem_number_map[5] = initial_value_helical_valley
    problem_number_map[6] = initial_value_powell_singular
    problem_number_map[7] = initial_value_freudenstein_roth
    problem_number_map[8] = initial_value_bard
    problem_number_map[9] = initial_value_kowalik_osborne
    problem_number_map[10] = initial_value_meyer
    problem_number_map[11] = initial_value_watson
    problem_number_map[12] = initial_value_box_3d
    problem_number_map[13] = initial_value_jennrich_sampson
    problem_number_map[14] = initial_value_brown_dennis
    problem_number_map[15] = initial_value_chebyquad
    problem_number_map[16] = initial_value_brown_almost_linear
    problem_number_map[17] = initial_value_osborne1
    problem_number_map[18] = initial_value_osborne2
    problem_number_map[19] = initial_value_bdqrtic
    problem_number_map[20] = initial_value_cube
    problem_number_map[21] = initial_value_mancino
    problem_number_map[22] = initial_value_heart8ls

    if prob_number not in problem_number_map:
        raise ValueError("Unknown problem number (must be in 1, ..., 22)")

    x0 = init_factor * problem_number_map[prob_number](n)

    return x0


########################################################
# Function 1 - Linear full rank
########################################################
def objective_linear_full_rank(x, n, m):
    assert (n >= 2), "n must be at least 2"  # check value of n
    assert (m >= n), "m must be at least n"  # check value of m
    fvec = np.zeros((m,))

    temp = 2.0 * np.sum(x) / float(m) + 1.0

    for i in range(m):
        fvec[i] = -temp
        if i < n:
            fvec[i] += x[i]

    return fvec


def initial_value_linear_full_rank(n):
    assert (n >= 2), "n must be at least 2"  # check value of n
    return np.ones((n,))


########################################################
# Function 2 - Linear rank 1
########################################################
def objective_linear_rank1(x, n, m):
    assert (n >= 2), "n must be at least 2"  # check value of n
    assert (m >= n), "m must be at least n"  # check value of m
    fvec = np.zeros((m,))

    temp = np.sum(x * np.arange(1, n + 1))  # sum i*x[i] for i=1,...,n

    for i in range(m):
        fvec[i] = float(i + 1) * temp - 1.0

    return fvec


def initial_value_linear_rank1(n):
    assert (n >= 2), "n must be at least 2"  # check value of n
    return np.ones((n,))


########################################################
# Function 3 - Linear rank 1 with zero columns and rows
########################################################
def objective_linear_rank1_with_zeros(x, n, m):
    assert (n >= 2), "n must be at least 2"  # check value of n
    assert (m >= n), "m must be at least n"  # check value of m
    fvec = np.zeros((m,))

    temp = np.sum(x[1:-1] * np.arange(2, n))  # sum i*x[i] for i=2,...,n-1

    for i in range(m - 1):
        fvec[i] = float(i) * temp - 1.0

    fvec[m - 1] = -1.0

    return fvec


def initial_value_linear_rank1_with_zeros(n):
    assert (n >= 2), "n must be at least 2"  # check value of n
    return np.ones((n,))


########################################################
# Function 4 - Rosenbrock function
########################################################
def objective_rosenbrock(x, n, m):
    assert (n == 2), "n must be 2"  # check value of n
    assert (m == 2), "m must be 2"  # check value of m
    fvec = np.zeros((m,))

    fvec[0] = 10.0 * (x[1] - x[0] ** 2)
    fvec[1] = 1.0 - x[0]

    return fvec


def initial_value_rosenbrock(n):
    assert (n == 2), "n must be 2"  # check value of n
    return np.array([-1.2, 1.0])


########################################################
# Function 5 - Helical valley function
########################################################
def objective_helical_valley(x, n, m):
    assert (n == 3), "n must be 3"  # check value of n
    assert (m == 3), "m must be 3"  # check value of m
    fvec = np.zeros((m,))

    if (x[0] > 0.0):
        th = atan(x[1] / x[0]) / (2.0 * pi)
    elif (x[0] < 0.0):
        th = atan(x[1] / x[0]) / (2.0 * pi) + 0.5
    else:  # x[0] == 0
        th = 0.25

    r = sqrt(x[0] ** 2 + x[1] ** 2)
    fvec[0] = 10.0 * (x[2] - 10.0 * th)
    fvec[1] = 10.0 * (r - 1.0)
    fvec[2] = x[2]

    return fvec


def initial_value_helical_valley(n):
    assert (n == 3), "n must be 3"  # check value of n
    return np.array([-1.0, 0.0, 0.0])


########################################################
# Function 6 - Powell singular function
########################################################
def objective_powell_singular(x, n, m):
    assert (n == 4), "n must be 4"  # check value of n
    assert (m == 4), "m must be 4"  # check value of m
    fvec = np.zeros((m,))

    fvec[0] = x[0] + 10.0 * x[1]
    fvec[1] = sqrt(5.0) * (x[2] - x[3])
    fvec[2] = (x[1] - 2.0 * x[2]) ** 2
    fvec[3] = sqrt(10.0) * (x[0] - x[3]) ** 2

    return fvec


def initial_value_powell_singular(n):
    assert (n == 4), "n must be 4"  # check value of n
    return np.array([3.0, -1.0, 0.0, 1.0])


########################################################
# Function 7 - Freudenstein and Roth function
########################################################
def objective_freudenstein_roth(x, n, m):
    assert (n == 2), "n must be 2"  # check value of n
    assert (m == 2), "m must be 2"  # check value of m
    fvec = np.zeros((m,))

    fvec[0] = -13.0 + x[0] + ((5.0 - x[1]) * x[1] - 2.0) * x[1]
    fvec[1] = -29.0 + x[0] + ((1.0 + x[1]) * x[1] - 14.0) * x[1]

    return fvec


def initial_value_freudenstein_roth(n):
    assert (n == 2), "n must be 2"  # check value of n
    return np.array([0.5, -2.0])


########################################################
# Function 8 - Bard function
########################################################
def objective_bard(x, n, m):
    assert (n == 3), "n must be 3"  # check value of n
    assert (m == 15), "m must be 15"  # check value of m
    fvec = np.zeros((m,))

    y1 = np.array([1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1, 3.9e-1, 3.7e-1, \
                   5.8e-1, 7.3e-1, 9.6e-1, 1.34e0, 2.1e0, 4.39e0])

    for i in range(1, 16):  # i = 1,..., 15
        tmp1 = float(i)
        tmp2 = float(16 - i)
        tmp3 = (tmp1 if i <= 8 else tmp2)
        fvec[i - 1] = y1[i - 1] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3))

    return fvec


def initial_value_bard(n):
    assert (n == 3), "n must be 3"  # check value of n
    return np.ones((n,))


########################################################
# Function 9 - Kowalik and Osborne function
########################################################
def objective_kowalik_osborne(x, n, m):
    assert (n == 4), "n must be 4"  # check value of n
    assert (m == 11), "m must be 11"  # check value of m
    fvec = np.zeros((m,))

    v = np.array([4.0e0, 2.0e0, 1.0e0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, \
                  7.14e-2, 6.25e-2])

    y2 = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, \
                   3.23e-2, 2.35e-2, 2.46e-2])

    for i in range(11):
        tmp1 = v[i] * (v[i] + x[1])
        tmp2 = v[i] * (v[i] + x[2]) + x[3]
        fvec[i] = y2[i] - x[0] * tmp1 / tmp2

    return fvec


def initial_value_kowalik_osborne(n):
    assert (n == 4), "n must be 4"  # check value of n
    return np.array([0.25, 0.39, 0.415, 0.39])


########################################################
# Function 10 - Meyer function
########################################################
def objective_meyer(x, n, m):
    assert (n == 3), "n must be 3"  # check value of n
    assert (m == 16), "m must be 16"  # check value of m
    fvec = np.zeros((m,))

    y3 = np.array([3.478e4, 2.861e4, 2.365e4, 1.963e4, 1.637e4, 1.372e4, 1.154e4, 9.744e3, \
                   8.261e3, 7.03e3, 6.005e3, 5.147e3, 4.427e3, 3.82e3, 3.307e3, 2.872e3])

    for i in range(1, 17):  # i = 1, ..., 16
        temp = 5.0 * float(i) + 45.0 + x[2]
        tmp1 = x[1] / temp

        try:
            tmp2 = exp(tmp1)
        except OverflowError:
            logging.warning("Overflow in Meyer, variable tmp2 for i = %i" % i)
            tmp2 = sys.float_info.max

        fvec[i - 1] = x[0] * tmp2 - y3[i - 1]

    return fvec


def initial_value_meyer(n):
    assert (n == 3), "n must be 3"  # check value of n
    return np.array([0.02, 4000.0, 250.0])


########################################################
# Function 11 - Watson function
########################################################
def objective_watson(x, n, m):
    assert (2 <= n <= 31), "n must be in 2, ..., 31"  # check value of n
    assert (m == 31), "m must be 31"  # check value of m
    fvec = np.zeros((m,))

    for i in range(1, 30):  # i=1,...,29
        div = float(i) / 29.0
        s1 = 0.0
        dx = 1.0
        for j in range(2, n + 1):  # j = 2,...,n
            s1 = s1 + (j - 1) * dx * x[j - 1]
            dx = div * dx
        s2 = 0.0
        dx = 1.0
        for j in range(1, n + 1):  # j = 1,...,n
            s2 = s2 + dx * x[j - 1]
            dx = div * dx
        fvec[i - 1] = s1 - s2 ** 2 - 1.0

    fvec[29] = x[0]
    fvec[30] = x[1] - x[0] ** 2 - 1.0

    return fvec


def initial_value_watson(n):
    assert (2 <= n <= 31), "n must be in 2, ..., 31"  # check value of n
    return 0.5 * np.ones((n,))


########################################################
# Function 12 - Box 3-dimensional function
########################################################
def objective_box_3d(x, n, m):
    assert (n == 3), "n must be 3"  # check value of n
    assert (m >= n), "m must be at least n"  # check value of m
    fvec = np.zeros((m,))

    for i in range(1, m + 1):  # i = 1,...,m
        temp = float(i)
        tmp1 = temp / 10.0
        fvec[i - 1] = exp(-tmp1 * x[0]) - exp(-tmp1 * x[1]) + (exp(-temp) - exp(-tmp1)) * x[2]

    return fvec


def initial_value_box_3d(n):
    assert (n == 3), "n must be 3"  # check value of n
    return np.array([0.0, 10.0, 20.0])


########################################################
# Function 13 - Jennrich and Sampson function
########################################################
def objective_jennrich_sampson(x, n, m):
    assert (n == 2), "n must be 2"  # check value of n
    assert (m >= n), "m must be at least n"  # check value of m
    fvec = np.zeros((m,))

    for i in range(1, m + 1):  # i = 1,...,m
        temp = float(i)
        fvec[i - 1] = 2.0 + 2.0 * temp - exp(temp * x[0]) - exp(temp * x[1])

    return fvec


def initial_value_jennrich_sampson(n):
    assert (n == 2), "n must be 2"  # check value of n
    return np.array([0.3, 0.4])


########################################################
# Function 14 - Brown and Dennis function
########################################################
def objective_brown_dennis(x, n, m):
    assert (n == 4), "n must be 4"  # check value of n
    assert (m >= n), "m must be at least n"  # check value of m
    fvec = np.zeros((m,))

    for i in range(1, m + 1):  # i = 1,...,m
        temp = float(i) / 5.0
        tmp1 = x[0] + temp * x[1] - exp(temp)
        tmp2 = x[2] + sin(temp) * x[3] - cos(temp)
        fvec[i - 1] = tmp1 ** 2 + tmp2 ** 2

    return fvec


def initial_value_brown_dennis(n):
    assert (n == 4), "n must be 4"  # check value of n
    return np.array([25.0, 5.0, -5.0, -1.0])


########################################################
# Function 15 - Chebyquad function
########################################################
def objective_chebyquad(x, n, m):
    assert (n >= 2), "n must be at least 2"  # check value of n
    assert (m >= n), "m must be at least n"  # check value of m
    fvec = np.zeros((m,))

    for j in range(n):
        t1 = 1.0
        t2 = 2.0 * x[j] - 1.0
        t = 2.0 * t2
        for i in range(m):
            fvec[i] += t2
            th = t * t2 - t1
            t1 = t2
            t2 = th

    for i in range(1, m + 1):  # i = 1,...,m:
        fvec[i - 1] = fvec[i - 1] / float(n)
        if i % 2 == 0:  # if i is even
            fvec[i - 1] += 1.0 / (float(i ** 2) - 1.0)

    return fvec


def initial_value_chebyquad(n):
    assert (n >= 2), "n must be at least 2"  # check value of n
    return np.arange(1, n + 1) / float(n + 1)


########################################################
# Function 16 - Brown almost-linear function
########################################################
def objective_brown_almost_linear(x, n, m):
    assert (n >= 2), "n must be at least 2"  # check value of n
    assert (m == n), "m must be n"  # check value of m
    fvec = np.zeros((m,))

    fvec[:-1] = x[:-1] + np.sum(x) - float(n + 1)
    fvec[-1] = np.prod(x) - 1.0

    return fvec


def initial_value_brown_almost_linear(n):
    assert (n >= 2), "n must be at least 2"  # check value of n
    return 0.5 * np.ones((n,))


########################################################
# Function 17 - Osborne 1 function
########################################################
def objective_osborne1(x, n, m):
    assert (n == 5), "n must be 5"  # check value of n
    assert (m == 33), "m must be 33"  # check value of m
    fvec = np.zeros((m,))

    y4 = np.array([8.44e-1, 9.08e-1, 9.32e-1, 9.36e-1, 9.25e-1, 9.08e-1, 8.81e-1, 8.5e-1, \
                   8.18e-1, 7.84e-1, 7.51e-1, 7.18e-1, 6.85e-1, 6.58e-1, 6.28e-1, 6.03e-1, \
                   5.8e-1, 5.58e-1, 5.38e-1, 5.22e-1, 5.06e-1, 4.9e-1, 4.78e-1, 4.67e-1, \
                   4.57e-1, 4.48e-1, 4.38e-1, 4.31e-1, 4.24e-1, 4.2e-1, 4.14e-1, 4.11e-1, 4.06e-1])

    for i in range(33):
        temp = 10.0 * float(i)
        try:
            tmp1 = exp(-x[3] * temp)
        except OverflowError:
            logging.warning("Overflow in Osborne1, variable tmp1 for i = %i" % i)
            tmp1 = sys.float_info.max

        try:
            tmp2 = exp(-x[4] * temp)
        except OverflowError:
            logging.warning("Overflow in Osborne1, variable tmp2 for i = %i" % i)
            tmp2 = sys.float_info.max

        fvec[i] = y4[i] - (x[0] + x[1] * tmp1 + x[2] * tmp2)

    return fvec


def initial_value_osborne1(n):
    assert (n == 5), "n must be 5"  # check value of n
    return np.array([0.5, 1.5, 1.0, 0.01, 0.02])


########################################################
# Function 18 - Osborne 2 function
########################################################
def objective_osborne2(x, n, m):
    assert (n == 11), "n must be 11"  # check value of n
    assert (m == 65), "m must be 65"  # check value of m
    fvec = np.zeros((m,))

    y5 = np.array([1.366e0, 1.191e0, 1.112e0, 1.013e0, 9.91e-1, 8.85e-1, 8.31e-1, 8.47e-1, \
                   7.86e-1, 7.25e-1, 7.46e-1, 6.79e-1, 6.08e-1, 6.55e-1, 6.16e-1, 6.06e-1, \
                   6.02e-1, 6.26e-1, 6.51e-1, 7.24e-1, 6.49e-1, 6.49e-1, 6.94e-1, 6.44e-1, \
                   6.24e-1, 6.61e-1, 6.12e-1, 5.58e-1, 5.33e-1, 4.95e-1, 5.0e-1, 4.23e-1, \
                   3.95e-1, 3.75e-1, 3.72e-1, 3.91e-1, 3.96e-1, 4.05e-1, 4.28e-1, 4.29e-1, \
                   5.23e-1, 5.62e-1, 6.07e-1, 6.53e-1, 6.72e-1, 7.08e-1, 6.33e-1, 6.68e-1, \
                   6.45e-1, 6.32e-1, 5.91e-1, 5.59e-1, 5.97e-1, 6.25e-1, 7.39e-1, 7.1e-1, \
                   7.29e-1, 7.2e-1, 6.36e-1, 5.81e-1, 4.28e-1, 2.92e-1, 1.62e-1, 9.8e-2, 5.4e-2])

    for i in range(65):
        temp = float(i) / 10.0
        try:
            tmp1 = exp(-x[4] * temp)
        except OverflowError:
            logging.warning("Overflow in Osborne2, variable tmp1 for i = %i" % i)
            tmp1 = sqrt(sys.float_info.max)

        try:
            tmp2 = exp(-x[5] * (temp - x[8]) ** 2)
        except OverflowError:
            logging.warning("Overflow in Osborne2, variable tmp2 for i = %i" % i)
            tmp2 = sqrt(sys.float_info.max)

        try:
            tmp3 = exp(-x[6] * (temp - x[9]) ** 2)
        except OverflowError:
            logging.warning("Overflow in Osborne2, variable tmp3 for i = %i" % i)
            tmp3 = sqrt(sys.float_info.max)

        try:
            tmp4 = exp(-x[7] * (temp - x[10]) ** 2)
        except OverflowError:
            logging.warning("Overflow in Osborne2, variable tmp4 for i = %i" % i)
            tmp4 = sqrt(sys.float_info.max)

        fvec[i] = y5[i] - (x[0] * tmp1 + x[1] * tmp2 + x[2] * tmp3 + x[3] * tmp4)

    return fvec


def initial_value_osborne2(n):
    assert (n == 11), "n must be 11"  # check value of n
    return np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 5.5])


########################################################
# Function 19 - Bdqrtic
########################################################
def objective_bdqrtic(x, n, m):
    assert (n >= 5), "n must be at least 5"  # check value of n
    assert (m == 2 * (n - 4)), "m must be 2(n-4)"  # check value of m
    fvec = np.zeros((m,))

    for i in range(1, n - 3):  # i = 1, ..., n-4
        fvec[i - 1] = -4.0 * x[i - 1] + 3.0
        fvec[n - 4 + i - 1] = x[i - 1] ** 2 + 2 * x[i] ** 2 + 3 * x[i + 1] ** 2 + 4 * x[i + 2] ** 2 + 5 * x[-1] ** 2

    return fvec


def initial_value_bdqrtic(n):
    assert (n >= 5), "n must be at least 5"  # check value of n
    return np.ones((n,))


########################################################
# Function 20 - Cube
########################################################
def objective_cube(x, n, m):
    assert (n >= 2), "n must be at least 2"  # check value of n
    assert (m == n), "m must be n"  # check value of m
    fvec = np.zeros((m,))

    fvec[0] = x[0] - 1.0
    fvec[1:] = 10.0 * (x[1:] - x[:-1] ** 3)

    return fvec


def initial_value_cube(n):
    assert (n >= 2), "n must be at least 2"  # check value of n
    return 0.5 * np.ones((n,))


########################################################
# Function 21 - Mancino
########################################################
def objective_mancino(x, n, m):
    assert (n >= 2), "n must be at least 2"  # check value of n
    assert (m == n), "m must be n"  # check value of m
    fvec = np.zeros((m,))

    for i in range(1, n + 1):  # i = 1, ..., n
        ss = 0.0
        for j in range(1, n + 1):  # j = 1 ,..., n
            v2 = sqrt(x[i - 1] ** 2 + float(i) / float(j))
            ss += v2 * (sin(log(v2)) ** 5 + cos(log(v2)) ** 5)
        fvec[i - 1] = 1400.0 * x[i - 1] + float(i - 50) ** 3 + ss

    return fvec


def initial_value_mancino(n):
    assert (n >= 2), "n must be at least 2"  # check value of n
    x0 = np.zeros((n,))

    for i in range(n):  # i = 0, ..., n-1
        ss = 0.0
        for j in range(n):  # j = 0, ..., n-1
            temp = sqrt(float(i + 1) / float(j + 1))
            ss += temp * (sin(log(temp)) ** 5 + cos(log(temp)) ** 5)
        x0[i] = -8.710996e-4 * (float(i + 1 - 50) ** 3 + ss)

    return x0


########################################################
# Function 22 - Heart8ls
########################################################
def objective_heart8ls(x, n, m):
    assert (n == 8), "n must be 8"  # check value of n
    assert (m == 8), "m must be 8"  # check value of m
    fvec = np.zeros((m,))

    fvec[0] = x[0] + x[1] + 0.69

    fvec[1] = x[2] + x[3] + 0.044

    fvec[2] = x[4] * x[0] + x[5] * x[1] - x[6] * x[2] - x[7] * x[3] + 1.57

    fvec[3] = x[6] * x[0] + x[7] * x[1] + x[4] * x[2] + x[5] * x[3] + 1.31

    fvec[4] = x[0] * (x[4] ** 2 - x[6] ** 2) - 2.0 * x[2] * x[4] * x[6] + \
              x[1] * (x[5] ** 2 - x[7] ** 2) - 2.0 * x[3] * x[5] * x[7] + 2.65

    fvec[5] = x[2] * (x[4] ** 2 - x[6] ** 2) + 2.0 * x[0] * x[4] * x[6] + \
              x[3] * (x[5] ** 2 - x[7] ** 2) + 2.0 * x[1] * x[5] * x[7] - 2.0

    fvec[6] = x[0] * x[4] * (x[4] ** 2 - 3.0 * x[6] ** 2) + \
              x[2] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2) + \
              x[1] * x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2) + \
              x[3] * x[7] * (x[7] ** 2 - 3.0 * x[5] ** 2) + 12.6

    fvec[7] = x[2] * x[4] * (x[4] ** 2 - 3.0 * x[6] ** 2) - \
              x[0] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2) + \
              x[3] * x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2) - \
              x[1] * x[7] * (x[7] ** 2 - 3.0 * x[5] ** 2) - 9.48

    return fvec


def initial_value_heart8ls(n):
    assert (n == 8), "n must be 8"  # check value of n
    return np.array([-0.3, -0.39, 0.3, -0.344, -1.2, 2.69, 1.59, -1.5])
