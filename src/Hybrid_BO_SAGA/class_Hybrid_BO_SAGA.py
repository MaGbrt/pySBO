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
from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.SBX import SBX
from Evolution.Two_Points import Two_Points
from Evolution.Polynomial import Polynomial
from Evolution.Gaussian import Gaussian
from Evolution.Elitist import Elitist


from Evolution_Controls.POV_EC import POV_EC
from Evolution_Controls.Distance_EC import Distance_EC
from Evolution_Controls.Dynamic_Inclusive_EC import Dynamic_Inclusive_EC

class MACE:
    def __init__(self, dim, batch_size, population):
        self._dim           = dim
        self._batch_size    = batch_size        
        self._population    = population
        
        self._lb = np.zeros(self._dim)
        self._ub = np.ones(self._dim)

    def SAGA_selection(self, model):
        # Operators
        select_op = Tournament(2)
        crossover_op_1 = SBX(0.9, 2)
        crossover_op_2 = Two_Points(0.9)
        mutation_op_1 = Polynomial(0.1, 20)
        mutation_op_2 = Gaussian(0.3, 1.0)
        replace_op = Elitist()
                
        
        # Evolution Controls
        ec_base_y = POV_EC(model)
        ec_base_d = Distance_EC(model)
        ec_op = Dynamic_Inclusive_EC(TIME_BUDGET, N_SIM, N_PRED, ec_base_d, ec_base_y)
        
        
        #----------------------Start Generation loop----------------------#
        for curr_gen in range(N_GEN):
            print("generation "+str(curr_gen))
        
            # Acquisition Process (SBX, Polynomial)
            t_AP_start = time.time()
            parents_1 = select_op.perform_selection(pop, N_CHLD//4)
            children = crossover_op_1.perform_crossover(parents_1)
            children = mutation_op_1.perform_mutation(children)
            del parents_1
            # Acquisition Process (SBX, Gaussian)
            parents_2 = select_op.perform_selection(pop, N_CHLD//4)
            children_2 = crossover_op_1.perform_crossover(parents_2)
            children_2 = mutation_op_2.perform_mutation(children_2)
            children.append(children_2)
            del children_2
            del parents_2
            # Acquisition Process (Two_Points, Polynomial)
            parents_3 = select_op.perform_selection(pop, N_CHLD//4)
            children_3 = crossover_op_2.perform_crossover(parents_3)
            children_3 = mutation_op_1.perform_mutation(children_3)
            children.append(children_3)
            del children_3
            del parents_3
            # Acquisition Process (Two_Points, Gaussian)
            parents_4 = select_op.perform_selection(pop, N_CHLD//4)
            children_4 = crossover_op_2.perform_crossover(parents_4)
            children_4 = mutation_op_2.perform_mutation(children_4)
            children.append(children_4)
            del children_4
            del parents_4
            assert p.is_feasible(children.dvec)
            t_AP_end = time.time()
            T_AP += (t_AP_end-t_AP_start)
        
            # Update nb_sim_per_proc (end of the search)
            t_now = time.time()
            elapsed_time = (t_now-t_start)+elapsed_sim_time
            remaining_time = TIME_BUDGET-elapsed_time
            if remaining_time<=SIM_TIME:
                break
            sim_afford = int(remaining_time//SIM_TIME)
            if np.max(nb_sim_per_proc)>sim_afford:
                nb_sim_per_proc = sim_afford*np.ones((nprocs,), dtype=int)
                SIM_TIME = 10000
            
            # Update dynamic EC
            t_AP_start = time.time()
            if isinstance(ec_op, Dynamic_Inclusive_EC):
                ec_op.update_active(elapsed_time)
        
            # Evolution Control
            idx_split = ec_op.get_sorted_indexes(children)
            to_simulate = Population(p)
            to_simulate.dvec = children.dvec[idx_split[0:np.sum(nb_sim_per_proc)]]
            del children
            t_AP_end = time.time()
            T_AP += (t_AP_end-t_AP_start)

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