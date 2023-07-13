#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:08:21 2023

@author: maxime
"""
import numpy as np
from platypus import NSGAII, MOEAD, Problem, Real, SPEA2, NSGAIII, Solution, InjectedPopulation, Archive
from math import pow, log, sqrt
from scipy.optimize import fmin_l_bfgs_b
import torch

class MACE:
    def __init__(self, dim, batch_size, mo_eval = 500):
        self._dim           = dim
        self._batch_size    = batch_size
        self._mo_eval       = mo_eval
        
        self._lb = np.zeros(self._dim)
        self._ub = np.ones(self._dim)


    def gen_guess(self, model):
        num_guess     = 10
        guess_x       = np.zeros((num_guess, self._dim))
        guess_x[0, :] = self._best_x
        self._m = model
        def obj(x):
            x = torch.tensor(x)
            m = self._m.predict(x[None, :])
            return m.detach().numpy()[0]
        def gobj(x):
            x = torch.tensor(x)
            dmdx = self._m.predictive_gradient(x[None, :])
            return dmdx[0].reshape(self._dim)

        bounds = [(self._lb[i], self._ub[i]) for i in range(self._dim)]
        for i in range(1, num_guess):
            xx = self._best_x + np.random.randn(self._best_x.size).reshape(self._best_x.shape) * 1e-3
            def mobj(x):
                return obj(x)
            def gmobj(x):
                return gobj(x).astype('float64')
            
            x, _, _ = fmin_l_bfgs_b(func=mobj, x0=xx, fprime=gmobj, bounds=bounds)
            # print('after l_bfgs:', x)
            guess_x[i, :] = np.array(x)
        return guess_x



    def generate_batch(self, model):
        i_min = torch.argmin(model._train_Y)
        self._best_x = model._train_X[i_min].detach().numpy()

        guess_x   = self.gen_guess(model)
        num_guess = guess_x.shape[0]
        # print('Guess x: ', guess_x)
        def obj(x):
            lcb, ei, pi = model.MACE_obj(np.array([x]))
            log_ei      = np.log(1e-40 + ei)
            log_pi      = np.log(1e-40 + pi)
            return [lcb, -1*log_ei, -1*log_pi]

        problem = Problem(self._dim, 3)
        for i in range(self._dim):
            problem.types[i] = Real(self._lb[i], self._ub[i])

        init_s = [Solution(problem) for i in range(num_guess)]
        for i in range(num_guess):
            init_s[i].variables = [x for x in guess_x[i, :]]

        problem.function = obj
        gen              = InjectedPopulation(init_s)
        arch             = Archive()
        algorithm        = NSGAII(problem, population = 100, generator = gen, archive = arch)
        def cb(a):
            print(a.nfe, len(a.archive), flush=True)
        algorithm.run(self._mo_eval)#, callback=cb)
        # print('Run MOO: DONE')
        # print(len(algorithm.result))
    
        if len(algorithm.result) > self._batch_size:
            optimized = algorithm.result
        else:
            optimized = algorithm.population
        # print('Get results: ', optimized)
        idxs = np.arange(len(optimized))
        idxs = np.random.permutation(idxs)
        idxs = idxs[0:self._batch_size]
        # print('Randomly select n_batch individuals')
        # print(idxs)
        batch = np.zeros((self._batch_size, self._dim))
        q=0
        for i in idxs:
            x = np.array(optimized[i].variables)
            batch[q] = x.reshape(1, self._dim)
            q += 1
        return torch.tensor(batch)