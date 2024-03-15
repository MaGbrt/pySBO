#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stolen from pySBO examples
"""
import os
import itertools
import copy
import numpy as np
import time
import torch
from mpi4py import MPI
from Evolution.Swarm import Swarm
from Evolution.Population import Population
from Evolution.Star_Clusters import Star_Clusters
from Evolution.Inertia import Inertia
from Evolution.Constriction import Constriction
from Evolution.Absorbent_Walls import Absorbent_Walls

from Surrogates.GP import GP

from Evolution_Controls.POV_EC import POV_EC
from Evolution_Controls.Distance_EC import Distance_EC
from Evolution_Controls.Dynamic_Inclusive_EC import Dynamic_Inclusive_EC

import Global_Var
from Global_Var import *

def par_SAPSO_SaaF_run(DB, n_cycle, t_max, batch_size, id_run = None, comm = None):
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:
        print('Saving as population of ', DB._size, ' individuals.')
        DB.save_as_population('SAPSO_SaaF' + id_run)
        time_per_cycle = np.zeros((n_cycle, 5))
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        
        # Arguments of the search
        SWARM_SIZE=32
        N_VEL = 128 # number of children issued per generation
        N_SIM = 32 # number of simulations per generation
        N_PRED = 0 # number of predictions per generation
        N_DISC = 96 # number of rejections per generation
        assert N_SIM+N_PRED+N_DISC==N_VEL
        assert N_SIM!=0

        N_GEN = int(n_cycle*nprocs/N_SIM) # If budget is given in cycles
        print('Budget of ', n_cycle, ' corresponds to ', N_GEN, ' generations.')

        TIME_BUDGET = t_max
        
        T_AP=0
        T_train=0
    
        # Files        
        try: 
            os.mkdir("./tmp")
        except: 
            print('Folder tmp already exists')
        try: 
            os.mkdir("./tmp/my_exp/")
        except: 
            print('Folder tmp/my_exp/ already exists')
        try:
            os.mkdir("./outputs_GP96/")
        except:
            print('Folder outputs_GP96 already exists')

        TMP_STORAGE="./tmp/my_exp"
        SUFFIX="_SAPSO_SaaF32_" + id_run + '_'
        F_SIM_ARCHIVE="/sim_archive"+SUFFIX+".csv"
        F_BEST_PROFILE="/best_profile"+SUFFIX+".csv"
        F_INIT_POP='./Save_as_pop/Pop_SAPSO_SaaF'+id_run+'.csv' 
        F_TRAIN_LOG="/training_log"+SUFFIX+".csv"
        F_TRAINED_MODEL="/trained_model"+SUFFIX
    
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
        assert p.is_feasible(swarm.dvec)
    
        # Number of simulations per proc
        nb_sim_per_proc = (N_SIM//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(N_SIM%nprocs):
            nb_sim_per_proc[i+1]+=1
    
        # Neighborhoods, Velocity updater, Position updater
        nbhs = [Star_Clusters(swarm, SWARM_SIZE), Star_Clusters(swarm, 8)]
        vel_upd_ops = [Inertia(), Constriction()]
        pos_upd_ops = [Absorbent_Walls()]
    
        # Start chronometer
        elapsed_sim_time = 0 # to stop the search (for a time budget of 5 sec)

        # Creating surrogate
        t_train_start = time.time()
        # surr = BNN_MCD(TMP_STORAGE+F_SIM_ARCHIVE, p, float('inf'), TMP_STORAGE+F_TRAIN_LOG, TMP_STORAGE+F_TRAINED_MODEL, 5)
        surr = GP(TMP_STORAGE+F_SIM_ARCHIVE, p, 96, TMP_STORAGE+F_TRAIN_LOG, TMP_STORAGE+F_TRAINED_MODEL, 'matern2.5')
        surr.perform_training()
        t_train_end = time.time()
        t_model_ = t_train_end  - t_train_start
        T_train+=(t_train_end-t_train_start)
    
        # Logging
        swarm.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP, T_train])
    
        # Evolution Controls
        ec_base_y = POV_EC(surr)
        ec_base_d = Distance_EC(surr)
        ec_op = Dynamic_Inclusive_EC(TIME_BUDGET, N_SIM, N_PRED, ec_base_d, ec_base_y)
    
        #----------------------Start Generation loop----------------------#
        itr = 0
        t_0 = time.time()

        for curr_gen in range(N_GEN):
            time_per_cycle[itr, 0] = t_model_
            t_start_cyc = time.time()

            # Create temporary swarms
            t_AP_start = time.time()
            tmp_swarms = []
            for i in range(len(nbhs)*len(vel_upd_ops)*len(pos_upd_ops)):
                tmp_swarms.append(copy.deepcopy(swarm))
    
            # For each neighborhoods, get the best neighbor's index per particle
            idx_best_nbs = []
            for nbh in nbhs:
                idx_best_nbs.append(nbh.idx_best_nb_per_part(swarm))
    
            # For each temporary swarm
            counter=0
            for idx_best_nb, vel_upd_op, pos_upd_op in itertools.product(idx_best_nbs, vel_upd_ops, pos_upd_ops):
                # Velocity update
                vel_upd_op.perform_velocities_update(tmp_swarms[counter], idx_best_nb)
                # Positions update
                pos_upd_op.perform_positions_update(tmp_swarms[counter])
                counter+=1
    
            elapsed_time = (time.time()-t_0)+elapsed_sim_time

            # Update dynamic EC
            if isinstance(ec_op, Dynamic_Inclusive_EC):
                ec_op.update_active(elapsed_time)
    
            # Retaining one candidate per particle
            # For each particle, create a temporary Population with the possible dvec
            for i in range(SWARM_SIZE):
                tmp_pop = Population(p)
                tmp_pop.dvec = np.zeros((len(tmp_swarms), p.n_dvar))
                for j,tmp_swarm in enumerate(tmp_swarms):
                    tmp_pop.dvec[j] = tmp_swarm.dvec[i]
                # Only select the best dvec (and associated velocity) for particle i according to EC
                idxs_sorted = ec_op.get_sorted_indexes(tmp_pop)
                swarm.dvec[i] = tmp_swarms[idxs_sorted[0]].dvec[i]
                swarm.velocities[i] = tmp_swarms[idxs_sorted[0]].velocities[i]
                del tmp_pop
            del tmp_swarms
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
            y_tmp = p.perform_real_evaluation(swarm.dvec[0:nb_sim_per_proc[0]])
            swarm.obj_vals[0:nb_sim_per_proc[0]] = y_tmp
            for i in range(1,nprocs): # receiving from workers
                swarm.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            swarm.dvec = swarm.dvec[:np.sum(nb_sim_per_proc)]
            DB.add(torch.tensor(DB.my_map(swarm.dvec)), torch.tensor(swarm.obj_vals))

            swarm.fitness_modes = True*np.ones(swarm.obj_vals.shape, dtype=bool)
            swarm.pbest_dvec=swarm.pbest_dvec[:np.sum(nb_sim_per_proc)]
            swarm.pbest_obj_vals=swarm.pbest_obj_vals[:np.sum(nb_sim_per_proc)]
            swarm.velocities=swarm.velocities[:np.sum(nb_sim_per_proc)]
            swarm.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE)
            t_eval_ = time.time() - t_eval_start 

            # Surrogate update
            t_train_start = time.time()
            surr.perform_training()
            t_train_end = time.time()
            T_train+=(t_train_end-t_train_start)
            t_model_ = time.time() - t_train_start

            # Logging
            swarm.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP, T_train])
            # Exit Generation loop if budget time exhausted
            t_end_cyc = time.time()
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time.time() - t_0
                elapsed_sim_time = 0 # already included
            else:
                t_current += t_end_cyc - t_start_cyc + Global_Var.sim_cost*np.max(nb_sim_per_proc)
                elapsed_sim_time += np.max(nb_sim_per_proc)*Global_Var.sim_cost # fictitious cost
   
            print("Alg. SAPSO_SaaF, generation ", curr_gen, " took --- %s seconds ---" % (t_end_cyc - t_start_cyc))
            print('Best known target is: ', torch.min(DB._y).numpy())
            print('t_current: ', t_current)
            print('real time is ', time.time() - t_0)
            #print('Size of DB: ', DB._size)
            time_per_cycle[itr, 1] = t_ap_
            time_per_cycle[itr, 2] = t_train_end - t_start_cyc # t_ap + t_model
            time_per_cycle[itr, 3] = t_eval_ # t_sim
            time_per_cycle[itr, 4] = t_end_cyc - t_start_cyc # t_tot
            target[0, itr+1] = torch.min(DB._y).numpy()
   
            itr += 1
            if (itr >= N_GEN or t_current > t_max):
                print('Time is up, SAPSO SaaF is done.')
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

