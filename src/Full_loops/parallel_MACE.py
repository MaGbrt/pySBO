#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:30:42 2023

@author: maxime
"""
import numpy as np
from time import time
import torch
import gpytorch
import Global_Var
from Global_Var import *

from Surrogates.GPyTorch_models import GP_model
from MACE.class_MACE import MACE
    

def par_MACE_run(DB, n_cycle, t_max, batch_size, id_run = None, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    if(my_rank == 0):
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))
        #print('DB\n', DB._X, DB._y)
        
        feval_budget = Global_Var.af_options['maxfun']
        mace = MACE(DB._dim, batch_size, feval_budget)
        
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
#        for i_cycle in range(n_cycle):
            t_start = time()
            # Fit a GP model
            scaled_y = DB.min_max_y_scaling()
            # print('Declare model')
            model = GP_model(DB._X, scaled_y)
    
            M1_time = time()
            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                # Fit the model
                # print('Fit model')
                model.custom_fit(Global_Var.large_fit)
                t_model = time ()
                time_per_cycle[iter, 0] = t_model - t_start
                
                ######################################################################
                # Create a batch
                # print('Generate batch')
                
                X_next = mace.generate_batch(model = model)
                # print('Candidates are:\n', X_next)
                t_ap = time()
                time_per_cycle[iter, 1] = t_ap - t_model
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
            print("Alg. MACE, cycle ", iter, " took --- %s seconds ---" % (M2_time - M1_time))
            print('Best known target is: ', torch.min(DB._y).numpy())

    
            # Print current status
            # print(f"{len(model._train_X)}) Best value: {Turbo._state.best_value:.2e}, TR length: {Turbo._state.length:.2e}")
            # print(torch.min(DB._y).numpy())
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 2] = t_ap - t_start # t_ap + t_model
            time_per_cycle[iter, 3] = t_end - t_ap # t_sim
            time_per_cycle[iter, 4] = t_end - t_start # t_tot
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)

#            t_current = time() - t_0
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)


        DB.print_min()
        return target, time_per_cycle

    else:
        for iter in range(n_cycle + 1):
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
                
        return None
                


# out_file = 'TuRBO_' + func + '_4D_rand' + str(n_init) + '_batch' + str(batch_size) + '_DoE' + str(DoE_num) + '.txt'
# np.savetxt(out_file, data, fmt='%.8e')


