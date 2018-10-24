#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from neldermead import NelderMead


def get_regular_simplex(dimension, centroid, gamma):
    angle = -1.0 / dimension
    regular_simplex = np.zeros([dimension+1, dimension])

    init_norm = 1
    tmp = 0.0
    for i in range(dimension):
        n_coord = np.sqrt((init_norm*init_norm) - tmp)
        regular_simplex[i][i] = n_coord
        r_coord = (angle - tmp) / n_coord
        for s in range(i+1, dimension+1):
            regular_simplex[s][i] = r_coord
        tmp += r_coord * r_coord
    for i in range(dimension+1):
        regular_simplex[i] *= gamma
    for i in range(dimension+1):
        regular_simplex[i] = (regular_simplex[i].reshape(dimension,1) - centroid).reshape(dimension,)
    return regular_simplex.T


def sphere(x):
    return np.sum(x**2)

def main():
    dim = 2
    iteration_number = 100
    f = sphere

    simplex = get_regular_simplex(dim, -np.zeros([dim, 1]), np.sqrt(2.0))
    nm = NelderMead(dim, f, simplex)
    x_best, f_best = nm.optimize(iteration_number)
    print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()
