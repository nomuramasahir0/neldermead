#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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
