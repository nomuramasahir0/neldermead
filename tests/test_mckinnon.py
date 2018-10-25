#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from neldermead import NelderMead
from scipy.optimize import minimize
import math

'''
    for reference.)
    K.I.M. McKinnon (1997), \Convergence of the Nelder{Mead simplex method to
    a nonstationary point", preprint (to be published in SIAM J. Optim.).
    
    the minimizer of this problem is [0, -1/2] and then the evaluation value is -1/4,
    but given certain init point, nelder-mead algorithm inevitably fail.
'''
def mckinnon(x):
    assert x.shape[0] == 2

    x = x.reshape(-1)
    g = x[1] + math.pow(x[1], 2)
    theta_small = 6
    theta_big = 360
    theta = theta_big if x[0] <= 0 else theta_small

    return theta * math.pow(x[0], 2) + g


def test_run_d2_mckinnon():
    print("test_run_d2:")
    dim = 2
    iteration_number = 100
    simplex = np.array([[0., 0.], [(1 + math.sqrt(33)) / 8, (1 - math.sqrt(33)) / 8], [1., 1.]]).T

    nm = NelderMead(dim, mckinnon, simplex)
    x_best, f_best = nm.optimize(iteration_number)
    print("f_best:{}".format(f_best))

    params = {'maxiter': iteration_number,
              'initial_simplex': simplex.T,
              'xatol': 1e-20,
              'fatol': 1e-20
              }
    scipy_nm_f_best = minimize(mckinnon, x0=np.zeros(dim), method='Nelder-Mead', options=params).fun

    print("scipy_f_best:{}".format(scipy_nm_f_best))

    assert abs(f_best - scipy_nm_f_best) < 1e-5
