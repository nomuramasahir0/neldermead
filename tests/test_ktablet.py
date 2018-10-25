#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from neldermead import NelderMead
from tests import utils
from scipy.optimize import minimize


def ktablet(x):
    x = x.reshape(-1)
    dim = len(x)
    k = int(dim / 4)
    return np.sum(x[0:k]**2) + np.sum((100*x[k:dim])**2)


def test_run_d10_ktablet():
    print("test_run_d10:")
    dim = 10
    iteration_number = 1000
    centroid = np.zeros([dim, 1])
    gamma = np.sqrt(dim)
    simplex = utils.get_regular_simplex(dim, centroid, gamma)

    nm = NelderMead(dim, ktablet, simplex)
    x_best, f_best = nm.optimize(iteration_number)
    print("f_best:{}".format(f_best))

    params = {'maxiter': iteration_number,
              'initial_simplex': simplex.T,
              'xatol': 1e-20,
              'fatol': 1e-20
              }
    scipy_nm_f_best = minimize(ktablet, x0=np.zeros(dim), method='Nelder-Mead', options=params).fun

    print("scipy_f_best:{}".format(scipy_nm_f_best))

    assert abs(f_best - scipy_nm_f_best) < 1e-5
