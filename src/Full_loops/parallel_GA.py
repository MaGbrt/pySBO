#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stolen from pySBO examples
"""

import os
import time
import torch
import numpy as np
from mpi4py import MPI

from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist

import Global_Var
from Global_Var import *


def par_GA_run(DB, n_cycle, t_max, id_run = None, comm = None):
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:
        print('Saving as population of ', DB._size, ' individuals.')
        DB.save_as_population('GA' + id_run)
        time_per_cycle = np.zeros((n_cycle, 5))
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()

        # Search arguments
        POP_SIZE = 32
        N_SIM = 32 # number of simulations per generation
        N_GEN = int(n_cycle*nprocs/N_SIM) # If budget is given in cycles
        print('Budget of ', n_cycle, ' corresponds to ', N_GEN, ' generations.')
        T_AP=0
        
        # Files
        try:
            os.mkdir("./tmp")
        except:
            print('Folder tmp already exists')
        try:
            os.mkdir("./tmp/my_exp/")
        except:
            print('Folder tmp/my_exp/ already exists')

        TMP_STORAGE="./tmp/my_exp"
        SUFFIX="_GA" + id_run + '_'
        F_SIM_ARCHIVE="/sim_archive"+SUFFIX+".csv"
        F_BEST_PROFILE="/best_profile"+SUFFIX+".csv"
        F_INIT_POP='./Save_as_pop/Pop_GA'+id_run+'.csv'
        
        # Population initialization and logging
        p = DB._obj
        pop = Population(p)
        pop.load_from_csv_file(F_INIT_POP)
        pop.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE)
        pop.dvec = pop.dvec[:POP_SIZE]
        pop.obj_vals = pop.obj_vals[:POP_SIZE]
        pop.fitness_modes = pop.fitness_modes[:POP_SIZE]
        pop.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP])
        assert p.is_feasible(pop.dvec)

        # Number of simulations per proc
        nb_sim_per_proc = (POP_SIZE//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(POP_SIZE%nprocs):
            nb_sim_per_proc[i+1]+=1
            
        # Operators
        select_op = Tournament(2)
        crossover_op = SBX(0.9, 2)
        mutation_op = Polynomial(0.1, 20)
        replace_op = Elitist()

        #----------------------Start Generation loop----------------------#
        itr = 0
        t_0 = time.time()

        for curr_gen in range(N_GEN):
            time_per_cycle[itr, 0] = 0
            t_start_cyc = time.time()

            # Acquisition Process
            t_AP_start = time.time()
            parents = select_op.perform_selection(pop, POP_SIZE)
            children = crossover_op.perform_crossover(parents)
            children = mutation_op.perform_mutation(children)
            assert p.is_feasible(children.dvec)
            t_AP_end = time.time()
            T_AP += (t_AP_end-t_AP_start)
            t_ap = (t_AP_end-t_AP_start)
  
            t_ap_ = time.time() - t_AP_start 
            t_eval_start = time.time()

           # Parallel simulations
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(children.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            children.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
            y_tmp = p.perform_real_evaluation(children.dvec[0:nb_sim_per_proc[0]])
            children.obj_vals[0:nb_sim_per_proc[0]] = y_tmp
            for i in range(1,nprocs): # receiving from workers
                children.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            children.dvec = children.dvec[:np.sum(nb_sim_per_proc)]
            DB.add(torch.tensor(DB.my_map(children.dvec)), torch.tensor(children.obj_vals))

            children.fitness_modes = True*np.ones(children.obj_vals.shape, dtype=bool)
            children.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE) # logging
            t_eval_ = time.time() - t_eval_start 

            # Replacement
            t_AP_start = time.time()
            replace_op.perform_replacement(pop, children)
            t_AP_end = time.time()
            T_AP += (t_AP_end-t_AP_start)
            t_ap += (t_AP_end-t_AP_start)

            assert p.is_feasible(pop.dvec)
            del children
            pop.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP])

            # Exit Generation loop if budget time exhausted
            t_end_cyc = time.time()
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time.time() - t_0
                elapsed_sim_time = 0 # already included
            else:
                t_current += t_end_cyc - t_start_cyc + Global_Var.sim_cost*np.max(nb_sim_per_proc)
                elapsed_sim_time += np.max(nb_sim_per_proc)*Global_Var.sim_cost # fictitious cost

            print("Alg. GA, generation ", curr_gen, " took --- %s seconds ---" % (t_end_cyc - t_start_cyc))
            print('Best known target is: ', torch.min(DB._y).numpy())
            print('t_current: ', t_current)
            print('real time is ', time.time() - t_0)
            #print('Size of DB: ', DB._size)
            time_per_cycle[itr, 1] = t_ap_
            time_per_cycle[itr, 2] = t_ap_ # t_ap + t_model
            time_per_cycle[itr, 3] = t_eval_ # t_sim
            time_per_cycle[itr, 4] = t_end_cyc - t_start_cyc # t_tot
            target[0, itr+1] = torch.min(DB._y).numpy()

            itr += 1
            if (itr >= N_GEN or t_current > t_max):
                print('Time is up, GA is done.')
                DB.print_min()
                print('DB has ', DB._size, 'points.')
                break
        #----------------------End Generation loop----------------------#
        # Stop workers
        print('Stop workers')
        for i in range(1,nprocs):
            comm.send(-1, dest=i, tag=10)
        return target, time_per_cycle

    #---------------------------------#
    #-------------WORKERS-------------#
    #---------------------------------#
    else:
        nsim = comm.recv(source=0, tag=10)
        p = DB._obj

        while nsim!=-1:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)
        else:
            print('Workers stopped')

        return None
