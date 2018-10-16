#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from neldermead.alg import NelderMead


def get_regular_simplex(dimension, centroid, gamma):
    angle = -1.0 / dimension
    regular_simplex = np.zeros([dimension+1, dimension])

    init_norm = 1
    tmp = 0.0
    for i in range(dimension):
        nCoord = np.sqrt((init_norm*init_norm) - tmp)
        regular_simplex[i][i] = nCoord
        rCoord = (angle - tmp) / nCoord

        for s in range(i+1, dimension+1):
            regular_simplex[s][i] = rCoord

        tmp += rCoord * rCoord
    # 最初のノルムをγとして, γ倍拡大
    # gamma = 1.0
    for i in range(dimension+1):
        regular_simplex[i] *= gamma

    for i in range(dimension+1):
        regular_simplex[i] = (regular_simplex[i].reshape(dimension,1) - centroid).reshape(dimension,)
    return regular_simplex.T


def sphere(x):
    return np.sum(x**2)


def main():
    dim = 3
    f = sphere
    simplex = get_regular_simplex(dim, -np.ones([dim, 1]) * 0.5, 0.4)
    nm = NelderMead(dim, f, simplex)

    x_best, f_best = nm.optimize(100)
    print("x_best:{}, f_best:{}".format(x_best, f_best))


if __name__ == '__main__':
    main()
