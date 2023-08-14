#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:59:45 2023

@author: maxime
"""


import numpy as np
from time import time
import torch
import gpytorch
import Global_Var
from Global_Var import *

from Surrogates.GPyTorch_models import GP_model
from TuRBO.TuRBO_class import TuRBO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
    
import os
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

def par_Hybrid_TuRBO_SAGA_run(DB, n_cycle, t_max, batch_size, threshold, id_run = None, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    
    if(my_rank == 0):
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))
        Turbo = TuRBO(DB._dim, batch_size)
        #print('DB\n', DB._X, DB._y)
        
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        
        itr = 0
        t_0 = time()
        print('Start timer: ', t_max, 's from this point')
        t_current = 0.
        while (itr < n_cycle and t_current < t_max and DB._size < threshold):
#        for i_cycle in range(n_cycle):
            t_start = time()
            # Fit a GP model
            Y_turbo = -DB._y.clone().detach()
            scaled_y = (Y_turbo - Y_turbo.min()) / (Y_turbo.max()-Y_turbo.min())

            model = GP_model(DB._X, scaled_y)
    
            M1_time = time()
            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                # Fit the model
                model.custom_fit(Global_Var.large_fit)
#                model.fit()
                t_model = time ()
                time_per_cycle[itr, 0] = t_model - t_start
                
                ######################################################################
                # Create a batch
                X_next = Turbo.generate_batch(mod = model, batch_size = batch_size, acqf = 'ei')
                #print('Candidates are:\n', X_next)
                t_ap = time()
                time_per_cycle[itr, 1] = t_ap - t_model
                ######################################################################
                
            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(X_next)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(X_next)):
                send_cand = X_next[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            y_new = np.zeros(n_cand[0])
            for c in range(int(n_cand[0])):
                y_new[c] = DB.eval_f(X_next[n_proc*c].numpy())

            ## Gather
            Y_next = torch.zeros(len(X_next), 1)
            k = 0
            for c in range(len(X_next)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(X_next[c].unsqueeze(0), torch.tensor(recv_eval))
                    Y_next[c] = torch.tensor(recv_eval)
                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
                else:
                    Y_next[c] = torch.tensor(y_new[k])
                    DB.add(X_next[c].unsqueeze(0), torch.tensor(y_new[k]).unsqueeze(0))
                    k =+ 1
            M2_time = time()
            
            # Update state
            Turbo.update_state(-Y_next)
            print("Alg. TuRBO, cycle ", itr, " took --- %s seconds ---" % (M2_time - M1_time))
            print('Best known target is: ', torch.min(DB._y).numpy())

    
            # Print current status
            # print(f"{len(model._train_X)}) Best value: {Turbo._state.best_value:.2e}, TR length: {Turbo._state.length:.2e}")
            # print(torch.min(DB._y).numpy())
            target[0, itr+1] = torch.min(DB._y).numpy()
            t_end = time()
            
            time_per_cycle[itr, 2] = t_ap - t_start # t_ap + t_model
            time_per_cycle[itr, 3] = t_end - t_ap # t_sim
            time_per_cycle[itr, 4] = t_end - t_start # t_tot
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[itr, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)

#            t_current = time() - t_0
            itr = itr + 1
        else :
            print('Budget for BO is out, time is %.6f' %t_current)
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)
            print('Saving as population of ', DB._size, ' individuals.')
            print('Threshold: ', threshold)
            DB.save_as_population(id_run)
        DB.print_min()
        
        
        # Switch to SAGA SaaF
        if my_rank==0:
            # Search arguments
            POP_SIZE = 32
            N_CHLD = 128 # number of children issued per generation
            N_SIM = 32 # number of simulations per generation
            N_PRED = 0 # number of predictions per generation
            N_DISC = 96 # number of rejections per generation
            assert N_SIM+N_PRED+N_DISC==N_CHLD
            assert N_SIM!=0
            budget = Global_Var.budget # number of cycles afordable 
            N_GEN = int((budget-itr+1)*n_proc/N_SIM)
            print('Maximum number of generations: ', N_GEN)
            TIME_BUDGET = Global_Var.max_time
            T_AP=0
            T_train=0
        
            # Files
            print(os.getcwd())
            
  
            try: 
                os.mkdir("./tmp")
            except: 
                print('Files already exist')
            try: 
                os.mkdir("./tmp/my_exp/")
            except: 
                print('Files already exist')

            TMP_STORAGE="./tmp/my_exp"
            SUFFIX="_Hybrid_TuRBO_SAGA_pop_"+id_run+"_"
            F_SIM_ARCHIVE="/sim_archive"+SUFFIX+".csv"
            F_BEST_PROFILE="/best_profile"+SUFFIX+".csv"
            F_INIT_POP= id_run # "Hybrid_TuRBO_SAGA_pop.csv" #  "./init_pop/init_pop_"+sys.argv[1]+"_"+sys.argv[2]+".csv"
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
            nb_sim_per_proc = (N_SIM//n_proc)*np.ones((n_proc,), dtype=int)
            for i in range(N_SIM%n_proc):
                nb_sim_per_proc[i+1]+=1
        
            # Operators
            select_op = Tournament(2)
            crossover_op_1 = SBX(0.9, 2)
            crossover_op_2 = Two_Points(0.9)
            mutation_op_1 = Polynomial(0.1, 20)
            mutation_op_2 = Gaussian(0.3, 1.0)
            replace_op = Elitist()
        
            # Start chronometer
            t_start = time()
            elapsed_sim_time = 0 # to stop the search (for a time budget of 5 sec)
        
            # Creating surrogate
            t_train_start = time()
            # surr = BNN_MCD(TMP_STORAGE+F_SIM_ARCHIVE, p, float('inf'), TMP_STORAGE+F_TRAIN_LOG, TMP_STORAGE+F_TRAINED_MODEL, 5)
            surr = GP(TMP_STORAGE+F_SIM_ARCHIVE, p, 96, TMP_STORAGE+F_TRAIN_LOG, TMP_STORAGE+F_TRAINED_MODEL, 'matern2.5')
            surr.perform_training()
            t_train_end = time()
            t_model_ = time() - t_train_start
            T_train+=(t_train_end-t_train_start)
        
            # Logging
            pop.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP, T_train])
            
            # Evolution Controls
            ec_base_y = POV_EC(surr)
            ec_base_d = Distance_EC(surr)
            ec_op = Dynamic_Inclusive_EC(TIME_BUDGET, N_SIM, N_PRED, ec_base_d, ec_base_y)
        
            #----------------------Start Generation loop----------------------#
            for curr_gen in range(N_GEN):
                time_per_cycle[itr, 0] = t_model_
                print("generation "+str(curr_gen))
                t_start_cyc = time()
                # Acquisition Process (SBX, Polynomial)
                t_AP_start = time()
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
                t_AP_end = time()
                T_AP += (t_AP_end-t_AP_start)

                # Update nb_sim_per_proc (end of the search)
                t_now = time()
                elapsed_time = (t_now-t_start)+elapsed_sim_time
                 
                # Update dynamic EC
                t_AP_start = time()
                if isinstance(ec_op, Dynamic_Inclusive_EC):
                    ec_op.update_active(elapsed_time)
        
                # Evolution Control
                idx_split = ec_op.get_sorted_indexes(children)
                to_simulate = Population(p)
                to_simulate.dvec = children.dvec[idx_split[0:np.sum(nb_sim_per_proc)]]
                del children
                t_AP_end = time()
                T_AP += (t_AP_end-t_AP_start)
                
                
                t_ap_ = time() - t_AP_start 

                t_eval_start = time()
                # print('To simulate: ', to_simulate.dvec)
                # Parallel simulations
                for i in range(1,n_proc): # sending to workers
                    comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                    comm.send(to_simulate.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
                to_simulate.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
                y_tmp = p.perform_real_evaluation(to_simulate.dvec[0:nb_sim_per_proc[0]])
                to_simulate.obj_vals[0:nb_sim_per_proc[0]] = y_tmp
#                print('Master simulates: ', torch.tensor(to_simulate.dvec[0:nb_sim_per_proc[0]]), torch.tensor(y_tmp))
                
                for i in range(1,n_proc): # receiving from workers
                    to_simulate.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
                to_simulate.dvec = to_simulate.dvec[:np.sum(nb_sim_per_proc)]
 #               print('Master receives from workers: ', to_simulate.dvec, to_simulate.obj_vals)
                DB.add(torch.tensor(DB.my_map(to_simulate.dvec)), torch.tensor(to_simulate.obj_vals))
                to_simulate.fitness_modes = True*np.ones(to_simulate.obj_vals.shape, dtype=bool)
                to_simulate.save_sim_archive(TMP_STORAGE+F_SIM_ARCHIVE)
        
                t_eval_ = time() - t_eval_start 
                # Surrogate update
                # print('# Surrogate update')
                t_train_start = time()
                surr.perform_training()
                t_train_end = time()
                T_train+=(t_train_end-t_train_start)
                t_model_ = time() - t_train_start
                
                # Replacement
                t_AP_start = time()
                replace_op.perform_replacement(pop, to_simulate)
                t_AP_end = time()
                T_AP += (t_AP_end-t_AP_start)
                assert p.is_feasible(pop.dvec)
                pop.update_best_sim(TMP_STORAGE+F_BEST_PROFILE, [T_AP, T_train])
                
                # Exit Generation loop if budget time exhausted
                t_end_cyc = time()
                if(Global_Var.sim_cost == -1): # real cost
                    t_current = time() - t_0
                    elapsed_sim_time = 0 # already included
                else:
                    t_current += t_end_cyc - t_start_cyc + Global_Var.sim_cost*np.max(nb_sim_per_proc)
                    elapsed_sim_time += np.max(nb_sim_per_proc)*Global_Var.sim_cost # fictitious cost

                print("Alg. SAGA, cycle ", itr, " took --- %s seconds ---" % (t_end_cyc - t_start_cyc))
                print('Best known target is: ', torch.min(DB._y).numpy())
                print('t_current: ', t_current)
                print('real time is ', time() - t_0)
                time_per_cycle[itr, 1] = t_ap_
                time_per_cycle[itr, 2] = t_train_end - t_start_cyc # t_ap + t_model
                time_per_cycle[itr, 3] = t_eval_ # t_sim
                time_per_cycle[itr, 4] = t_end_cyc - t_start_cyc # t_tot
                itr += 1
                if (itr > n_cycle or t_current > t_max):
                    print('Time is up, SAGA SaaF is done.')
                    DB.print_min()
                    break
            #----------------------End Generation loop----------------------#
        
            
            # Stop workers
            for i in range(1,n_proc):
                comm.send(-1, dest=i, tag=10)
        
        return target, time_per_cycle

    else:
        for itr in range(n_cycle + 1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')

            comm.Bcast(n_cand, root = 0)
            #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')
            if (n_cand.sum() == 0):
                break
            else :
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = my_rank + c * n_proc))
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
        
        
        p = DB._obj
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-1:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)

        return None
