#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stolen from pySBO examples
"""

import os
import sys
import numpy as np
import time
import torch
from mpi4py import MPI

from Evolution.Swarm import Swarm
from Evolution.Population import Population
from Evolution.Star_Clusters import Star_Clusters
from Evolution.Constriction import Constriction
from Evolution.Absorbent_Walls import Absorbent_Walls
import Global_Var
from Global_Var import *

def par_PSO_run(DB, n_cycle, t_max, id_run = None, comm = None):
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    
    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#        

    
    if rank==0:       
        print('Saving as population of ', DB._size, ' individuals.')
        DB.save_as_population('PSO' + id_run)
        time_per_cycle = np.zeros((n_cycle, 5))
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()

        # Arguments of the search
        SWARM_SIZE=32
        N_NBH=SWARM_SIZE
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
        SUFFIX="_PSO" + id_run + '_'
        F_SIM_ARCHIVE="/sim_archive"+SUFFIX+".csv"
        F_BEST_PROFILE="/best_profile"+SUFFIX+".csv"
        F_INIT_POP='./Save_as_pop/Pop_PSO'+id_run+'.csv'

        # Swarm initialization and logging
        p = DB._obj
        swarm = Swarm(p)
        Population.load_from_csv_file(swarm, F_INIT_POP)
        swarm.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE)
        swarm.dvec = swarm.dvec[:SWARM_SIZE]
        swarm.obj_vals = swarm.obj_vals[:SWARM_SIZE]
        swarm.fitness_modes = swarm.fitness_modes[:SWARM_SIZE]
        swarm.pbest_dvec = np.copy(swarm.dvec)
        swarm.pbest_obj_vals = np.copy(swarm.obj_vals)
        p_bounds = p.get_bounds()
        swarm.velocities = np.random.uniform(low=p_bounds[0,:], high=p_bounds[1,:], size=swarm.dvec.shape)
        del p_bounds
        swarm.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP])
        assert p.is_feasible(swarm.dvec)

        # Number of simulations per proc
        nb_sim_per_proc = (SWARM_SIZE//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(SWARM_SIZE%nprocs):
            nb_sim_per_proc[i+1]+=1

        # Neighborhood
        nbh = Star_Clusters(swarm, N_NBH)
        # Velocity updater
        vel_upd_op = Constriction(2.05, 2.05, 0.5)
        # Position updater
        pos_upd_op = Absorbent_Walls()

        #----------------------Start Generation loop----------------------#
        itr = 0
        t_0 = time.time()
        for curr_gen in range(N_GEN):
            time_per_cycle[itr, 0] = 0
            t_start_cyc = time.time()

            # Acquisition Process (update velocity and position)
            t_AP_start = time.time()
            idx_best_nb = nbh.idx_best_nb_per_part(swarm)
            vel_upd_op.perform_velocities_update(swarm, idx_best_nb)
            pos_upd_op.perform_positions_update(swarm)
            assert p.is_feasible(swarm.dvec)
            t_AP_end = time.time()
            T_AP += (t_AP_end-t_AP_start)

            t_ap_ = time.time() - t_AP_start 
            t_eval_start = time.time()            
            
            # Parallel evaluations
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(swarm.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            swarm.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
            swarm.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(swarm.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                swarm.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            swarm.dvec = swarm.dvec[:np.sum(nb_sim_per_proc)]
            DB.add(torch.tensor(DB.my_map(swarm.dvec)), torch.tensor(swarm.obj_vals))
            swarm.fitness_modes = True*np.ones(swarm.obj_vals.shape, dtype=bool)
            swarm.pbest_dvec=swarm.pbest_dvec[:np.sum(nb_sim_per_proc)]
            swarm.pbest_obj_vals=swarm.pbest_obj_vals[:np.sum(nb_sim_per_proc)]
            swarm.velocities=swarm.velocities[:np.sum(nb_sim_per_proc)]
            t_eval_ = time.time() - t_eval_start 

           # Logging
            swarm.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE)
            swarm.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP])

           # Exit Generation loop if budget time exhausted
            t_end_cyc = time.time()
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time.time() - t_0
                elapsed_sim_time = 0 # already included
            else:
                t_current += t_end_cyc - t_start_cyc + Global_Var.sim_cost*np.max(nb_sim_per_proc)
                elapsed_sim_time += np.max(nb_sim_per_proc)*Global_Var.sim_cost # fictitious cost

            print("Alg. PSO, generation ", curr_gen, " took --- %s seconds ---" % (t_end_cyc - t_start_cyc))
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
                print('Time is up, PSO is done.')
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
