#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class RealSolution(object):
    def __init__(self, dim):
        self.f = float('nan')
        self.x = np.zeros([dim, 1])


class NelderMead:
    def __init__(self, dim, f, simplex, **kwargs):
        self.dim = dim
        self.f = f
        self.simplex = [RealSolution(self.dim) for i in range(self.dim + 1)]
        self.no_of_evals = 0
        for i in range(self.dim + 1):
            self.simplex[i].x = simplex[:, i].reshape(dim, 1)
            self.simplex[i].f = self.f(self.simplex[i].x)
            self.no_of_evals += 1

        self.delta_e = kwargs.get('delta_e', 2.0)
        self.delta_oc = kwargs.get('delta_oc', 1 / 2.0)
        self.delta_ic = kwargs.get('delta_ic', -1 / 2.0)
        self.gamma = kwargs.get('gamma', 1 / 2.0)
        self.constraint = kwargs.get('constraint', [[- np.inf, np.inf] for _ in range(dim)])
        self.penalty_coef = kwargs.get('penalty_coef', 1e5)

        self.g = 0

        self.simplex.sort(key=lambda s: s.f)
        self.f_best = self.simplex[0].f
        self.x_best = np.empty(self.dim)

    def add_constraint_violation(self):
        for s in self.simplex:
            constraint_violation = 0.
            for j in range(self.dim):
                constraint_violation += (-min(0, s.x[j] - self.constraint[j][0]) + max(0, s.x[j] - self.constraint[j][1])) * self.penalty_coef
            s.f = s.f + constraint_violation

    def optimize(self, iterations):
        for i in range(iterations):
            _ = self.one_iteration()
        return self.x_best, self.f_best

    def one_iteration(self):
        self.g += 1
        fval_list = []
        x_list = []
        # xc
        xc = np.sum(np.array([self.simplex[i].x for i in range(self.dim)])) / self.dim
        # xr
        xr = RealSolution(self.dim)
        xr.x = xc + (xc - self.simplex[self.dim].x)
        xr.f = self.f(xr.x)
        self.no_of_evals += 1
        fval_list.append(xr.f)
        x_list.append(xr.x)

        # i.
        if xr.f < self.f_best:
            xe = RealSolution(self.dim)
            xe.x = xc + self.delta_e * (xc - self.simplex[self.dim].x)
            xe.f = self.f(xe.x)
            self.no_of_evals += 1
            fval_list.append(xe.f)
            x_list.append(xe.x)
            if xr.f < xe.f:
                self.simplex[self.dim] = xr
            else:
                self.simplex[self.dim] = xe
        elif self.f_best <= xr.f and xr.f < self.simplex[self.dim - 1].f:
            self.simplex[self.dim] = xr
        elif self.simplex[self.dim - 1].f <= xr.f and xr.f < self.simplex[self.dim].f:
            xoc = RealSolution(self.dim)
            xoc.x = xc + self.delta_oc * (xc - self.simplex[self.dim].x)
            xoc.f = self.f(xoc.x)
            self.no_of_evals += 1
            fval_list.append(xoc.f)
            x_list.append(xoc.x)
            if xr.f < xoc.f:
                self.simplex[self.dim] = xr
            else:
                self.simplex[self.dim] = xoc
        else:
            xic = RealSolution(self.dim)
            xic.x = xc + self.delta_ic * (xc - self.simplex[self.dim].x)
            xic.f = self.f(xic.x)
            self.no_of_evals += 1
            fval_list.append(xic.f)
            x_list.append(xic.x)
            if xic.f < self.simplex[self.dim].f:
                self.simplex[self.dim] = xic
            else:
                for i in range(self.dim):
                    self.simplex[i+1].x = self.simplex[0].x + self.gamma * (self.simplex[i+1].x - self.simplex[0].x)
                    self.simplex[i+1].f = self.f(self.simplex[i+1].x)
                    self.no_of_evals += 1
                    fval_list.append(self.simplex[i+1].f)
                    x_list.append(self.simplex[i+1].x)

        evals_no_penalty = [self.simplex[i].f for i in range(len(self.simplex))]
        best_index = int(np.argmin(evals_no_penalty))

        if self.simplex[best_index].f < self.f_best:
            self.f_best = self.simplex[best_index].f
            self.x_best = self.simplex[best_index].x

        self.add_constraint_violation()
        self.simplex.sort(key=lambda s: s.f)

        return x_list, fval_list
