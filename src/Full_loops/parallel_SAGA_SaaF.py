#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stolen from pySBO examples
"""

import os
import sys
import time
import numpy as np
import torch 


from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.SBX import SBX
from Evolution.Two_Points import Two_Points
from Evolution.Polynomial import Polynomial
from Evolution.Gaussian import Gaussian
from Evolution.Elitist import Elitist

from Surrogates.GP import GP

from Evolution_Controls.POV_EC import POV_EC
from Evolution_Controls.Distance_EC import Distance_EC
from Evolution_Controls.Dynamic_Inclusive_EC import Dynamic_Inclusive_EC

import Global_Var
from Global_Var import *
dtype = torch.double
    


def par_SAGA_SaaF_run(DB, n_cycle, t_max, batch_size, id_run = None, comm = None):
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:
        print('Saving as population of ', DB._size, ' individuals.')
        DB.save_as_population('SAGA_SaaF' + id_run)
        time_per_cycle = np.zeros((n_cycle, 5))
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()


        # Search arguments
        POP_SIZE = 32
        N_CHLD = 128 # number of children issued per generation
        N_SIM = 32 # number of simulations per generation
        N_PRED = 0 # number of predictions per generation
        N_DISC = 96 # number of rejections per generation
        assert N_SIM+N_PRED+N_DISC==N_CHLD
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
        SUFFIX="_SAGA_SaaF32"+id_run+"_" #str(nprocs)+"_"+sys.argv[1]+"_"+sys.argv[2]
        F_SIM_ARCHIVE="/sim_archive"+SUFFIX+".csv"
        F_BEST_PROFILE="/best_profile"+SUFFIX+".csv"
        F_INIT_POP='./Save_as_pop/Pop_SAGA_SaaF'+id_run+'.csv' #"./init_pop/init_pop_"+sys.argv[1]+"_"+sys.argv[2]+".csv"
        F_TRAIN_LOG="/training_log"+SUFFIX+".csv"
        F_TRAINED_MODEL="/trained_model"+SUFFIX


        # Population initialization and logging
        p = DB._obj
        pop = Population(p)
        pop.load_from_csv_file(F_INIT_POP)
        pop.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE)
        pop.dvec = pop.dvec[:POP_SIZE]
        pop.obj_vals = pop.obj_vals[:POP_SIZE]
        pop.fitness_modes = pop.fitness_modes[:POP_SIZE]
        assert p.is_feasible(pop.dvec)

        # Number of simulations per proc
        nb_sim_per_proc = (N_SIM//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(N_SIM%nprocs):
            nb_sim_per_proc[i+1]+=1

        # Operators
        select_op = Tournament(2)
        crossover_op_1 = SBX(0.9, 2)
        crossover_op_2 = Two_Points(0.9)
        mutation_op_1 = Polynomial(0.1, 20)
        mutation_op_2 = Gaussian(0.3, 1.0)
        replace_op = Elitist()

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
        pop.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP, T_train])
        
        # Evolution Controls
        ec_base_y = POV_EC(surr)
        ec_base_d = Distance_EC(surr)
        ec_op = Dynamic_Inclusive_EC(TIME_BUDGET, N_SIM, N_PRED, ec_base_d, ec_base_y)

        
        #----------------------Start Generation loop----------------------#
        itr = 0
        t_0 = time.time()

        for curr_gen in range(N_GEN):
            time_per_cycle[itr, 0] = t_model_
#            print('Time to fit the model: ', time_per_cycle[itr, 0])
            t_start_cyc = time.time()

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
            elapsed_time = (t_now-t_0)+elapsed_sim_time
                        
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
            t_ap = (t_AP_end-t_AP_start)
  
            t_ap_ = time.time() - t_AP_start 
            t_eval_start = time.time()

            # Parallel simulations
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(to_simulate.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            to_simulate.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
            y_tmp = p.perform_real_evaluation(to_simulate.dvec[0:nb_sim_per_proc[0]])
            to_simulate.obj_vals[0:nb_sim_per_proc[0]] = y_tmp
            for i in range(1,nprocs): # receiving from workers
                to_simulate.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            to_simulate.dvec = to_simulate.dvec[:np.sum(nb_sim_per_proc)]
            DB.add(torch.tensor(DB.my_map(to_simulate.dvec)), torch.tensor(to_simulate.obj_vals))

            to_simulate.fitness_modes = True*np.ones(to_simulate.obj_vals.shape, dtype=bool)
            to_simulate.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE)
            
            t_eval_ = time.time() - t_eval_start 
            
            # Surrogate update
            t_train_start = time.time()
            surr.perform_training()
            t_train_end = time.time()
            T_train+=(t_train_end-t_train_start)
            t_model_ = time.time() - t_train_start

            # Replacement
            t_AP_start = time.time()
            replace_op.perform_replacement(pop, to_simulate)
            t_AP_end = time.time()
            T_AP += (t_AP_end-t_AP_start)
            t_ap += (t_AP_end-t_AP_start)
            assert p.is_feasible(pop.dvec)
            pop.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP, T_train])
            
            # Exit Generation loop if budget time exhausted
            t_end_cyc = time.time()
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time.time() - t_0
                elapsed_sim_time = 0 # already included
            else:
                t_current += t_end_cyc - t_start_cyc + Global_Var.sim_cost*np.max(nb_sim_per_proc)
                elapsed_sim_time += np.max(nb_sim_per_proc)*Global_Var.sim_cost # fictitious cost

            print("Alg. SAGA_SaaF, generation ", curr_gen, " took --- %s seconds ---" % (t_end_cyc - t_start_cyc))
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
                print('Time is up, SAGA SaaF is done.')
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

