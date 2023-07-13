#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:53:10 2022

@author: maxime
"""

## EGO Algorithm using BOTorch framework
import numpy as np
import torch
import gpytorch
import Global_Var
from Global_Var import *

from Surrogates.GPyTorch_models import GP_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import qExpectedImprovement
from BSP_tree.tree import Tree
from BSP_tree.subdomains import Subdomains
from BSP_tree.split_functions import default_split
from time import time

from botorch.optim import optimize_acqf
dtype = torch.double
max_cholesky_size = float("inf")  # Always use Cholesky

#%% 
## par_g1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    # BSP EGO with one global model, both AP and evaluations are parallel
  
## batch_par_g1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    # BSP EGO with one global model, only evaluation of objective function is parallel
    
## par_g1_BSP2_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    # No limit for number of leaves, but not all leaves are activated
    
## par_g1_BSP_qEGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    # Use of MCbased qEGO inside each subregion


#%%
def par_g1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    dim = DB._dim
    tol = 0.01# tolerance/min distance

    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)

        subdomains = T.get_leaves() # Returns an element of class Subdomains

        T.check_volume()
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_current = 0.
        t_0 = time()
        while (iter < n_cycle and t_current < t_max):
#        for iter in range(n_cycle):
            keep_going = np.ones(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

            t_start = time()
            # Define and fit model
            #print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(n_leaves):
                send_to = c%n_proc
                n_tasks[send_to] += 1
            #print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)
            comm.Bcast(n_tasks, root = 0)

            # 2. Master sends the bounds to workers - Dispatch tasks (APs)
            my_tasks = n_tasks[my_rank]
            b_list = []
            for c in range(n_leaves):
                send_to = c%n_proc
                if send_to == 0:
                    bounds = subdomains._list[c]._domain
                    b_list.append(bounds)
                else:
                    bounds = subdomains._list[c]._domain
                    comm.send(bounds, dest = send_to, tag = c)

            # Create global model
            scaled_y = DB.min_max_y_scaling()
#                scaled_y = DB_temp.normal_y_scaling()            
            t_mod0 = time()
            model = GP_model(DB._X, scaled_y)
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                #model.fit()
                model.custom_fit(Global_Var.large_fit)  
                time_per_cycle[iter, 0] = time() - t_mod0
                crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)

            t_ap0 = time()
            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            for t in range(my_tasks):
                bounds = b_list[t]
                try :
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                except :
                    print('\n Failed to optimize the acquisition function. Why ?')
                    #DB_temp.try_distance(tol)
                    print('try again with new model an increase of the jitter')
                    break

                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
            #print('I (proc ', my_rank, ') propose candidates ', candidates)

            # 4. Master yields the candidates from workers
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
                get_from = c%n_proc
                if (get_from == 0):
                    candidates_list.append(candidates[cpt, 0:dim])
                    acq_value_list.append(candidates[cpt, dim])
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)
                    candidates_list.append(cand[0:dim])
                    tmp_crit = crit(torch.tensor(cand[0:dim]).unsqueeze(0))
                    acq_value_list.append(tmp_crit.detach().numpy()[0]) # eval_f crit with AF model from master
                subdomains._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
              
            #print('Choosing ', batch_size, ' candidates among ', len(sort_X))
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            t_ap = time()
            time_per_cycle[iter, 1] = time() - t_ap0
            time_per_cycle[iter, 2] = t_ap - t_start
        
            ######################################################################
            
            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])
                x_new = selected_candidates[n_proc*c] 
#                print(torch.tensor(x_new).unsqueeze(0))
                DB.add(torch.tensor(x_new, dtype = torch.float64).unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(torch.tensor(selected_candidates[c]).unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            T.update(subdomains)

            target[0, iter+1] = torch.min(DB._y).numpy()
            
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            print("Alg. g1_BSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        return target, time_per_cycle
    
    else: # workers
        init_size = DB._size      
        
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            if(keep_going.sum() == 1):    
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                
                #print('I (proc ', my_rank, ') received the DB')
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
                
                DB.set_Xy(tmp_X, tmp_y)

                # 1. Workers receive the number of leaves to explore ; i.e number of AP to perform
                n_tasks = np.zeros(n_proc, dtype = 'i')
                comm.Bcast(n_tasks, root = 0)
                #print('I (proc ', my_rank, ') am receiving ', n_tasks, ' tasks')
    
                # 2. Workers receive the bounds to perform APs   
                my_tasks = n_tasks[my_rank] 
                b_list = []
                bounds = np.zeros((2, DB._dim))
                for t in range(my_tasks):
                    bounds = comm.recv(source = 0, tag = my_rank + t * n_proc)
                    b_list.append(bounds)
#                print('I need to perform an AP in each subdomain of the following list \n', b_list)
                
                # Create global model
                scaled_y = DB.min_max_y_scaling()
    #                scaled_y = DB_temp.normal_y_scaling()            
                model = GP_model(DB._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)

                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB._dim + 1))

                for t in range(my_tasks):
                    bounds = b_list[t]
                    try :
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        break

                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                #print('I (proc ', my_rank, ') propose candidates ', candidates)

                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
    
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

                cand = []
                for c in range(n_cand[my_rank]):
                    temp_cand = comm.recv(source = 0, tag = my_rank + c * n_proc)
                    cand.append(temp_cand)
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
            
            ## Receive from master 
            else :
                print('No more candidate, break')
                break
            
        return None        

#%%
def batch_par_g1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()

    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)

        subdomains = T.get_leaves() # Returns an element of class Subdomains

        T.check_volume()
        
        time_per_cycle = np.zeros((n_cycle, 4))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
#        for iter in range(n_cycle):
            t_start = time()
            # Define and fit model
            # Output data scaled in [0, 1] - min_max_scaling
            scaled_y = DB.min_max_y_scaling()
            model = GP_model(DB._X, scaled_y)
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                #model.custom_fit(200)
                model.fit()
                t_model = time ()
                time_per_cycle[iter, 0] = t_model - t_start
               #print("Model learning took --- %s seconds ---" % (t_model - t_start))
            
                ######################################################################
                # Acquisition process
                subdomains = T.get_leaves() # Returns an element of class Subdomains
                n_leaves = subdomains.get_size()
                sorted_candidates = subdomains.local_APs(n_cand = batch_size, model = model)
                candidates = DB.select_clean(sorted_candidates, batch_size, n_leaves, 0.001)
    
                #print('Candidates are:\n', candidates)
                t_ap = time()
                time_per_cycle[iter, 1] = t_ap - t_model

            ######################################################################
            
            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(candidates)):
                send_cand = candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(candidates[n_proc*c])
                x_new = candidates[n_proc*c] 
#                print(torch.tensor(x_new).unsqueeze(0))
                DB.add(torch.tensor(x_new, dtype = torch.float64).unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(torch.tensor(candidates[c]).unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            T.update(subdomains)

            target[0, iter+1] = torch.min(DB._y).numpy()
            
            t_end = time()
            time_per_cycle[iter, 2] = t_end - t_ap
            time_per_cycle[iter, 3] = t_end - t_start
            print("Alg. batch g1_BSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            iter = iter + 1
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
        else :
            print('Budget is out, time is %.6f' %t_current)
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)

        DB.print_min()
        return target, time_per_cycle
    else:
        for iter in range(n_cycle+1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')

            comm.Bcast(n_cand, root = 0)
            #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

            ## Receive from master 
            if (n_cand.sum() == 0):
                break
            else :
                    cand = []
                    for c in range(n_cand[my_rank]):
                        temp_cand = comm.recv(source = 0, tag = my_rank + c * n_proc)
                        cand.append(temp_cand)
        
                    ## eval_f
                    y_new = []
                    for c in range(n_cand[my_rank]):
                        y_new.append(DB.eval_f(cand[c]))
        
                    ## Send it back
                    for c in range(n_cand[my_rank]):
                        comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
            
        return None        
        
    
#%%  
def par_g1_BSP_qEGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    dim = DB._dim
    tol = 0.01# tolerance/min distance
    
    n_cand_leaf = 2
    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)

        subdomains = T.get_leaves() # Returns an element of class Subdomains

        T.check_volume()
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_current = 0.
        t_0 = time()
        while (iter < n_cycle and t_current < t_max):
#        for iter in range(n_cycle):
            keep_going = np.ones(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

            t_start = time()
            Y_neg = -DB._y.clone().detach() # need to take opposite sign to perform maximization (pb with qEI when minimizing)

            # Define and fit model
            #print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(-DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(n_leaves):
                send_to = c%n_proc
                n_tasks[send_to] += 1
            #print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)
            comm.Bcast(n_tasks, root = 0)

            # 2. Master sends the bounds to workers - Dispatch tasks (APs)
            my_tasks = n_tasks[my_rank]
            b_list = []
            for c in range(n_leaves):
                send_to = c%n_proc
                if send_to == 0:
                    bounds = subdomains._list[c]._domain
                    b_list.append(bounds)
                else:
                    bounds = subdomains._list[c]._domain
                    comm.send(bounds, dest = send_to, tag = c)

            # Create global model
            scaled_y = (Y_neg - Y_neg.min()) / (Y_neg.max()-Y_neg.min())
#                scaled_y = DB_temp.normal_y_scaling()            
            t_mod0 = time()
            model = GP_model(DB._X, scaled_y)
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                #model.fit()
                model.custom_fit(Global_Var.large_fit)  
                time_per_cycle[iter, 0] = time() - t_mod0
                crit = qExpectedImprovement(model._model, torch.max(model._train_Y))

            t_ap0 = time()
            # 3. Perform AP for each subdomain
            candidates = np.zeros((n_cand_leaf*my_tasks, DB._dim + 1))
            for t in range(my_tasks):
                bounds = b_list[t]
                try :
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=n_cand_leaf, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                except :
                    print('\n Failed to optimize the acquisition function. Why ?')
                    #DB_temp.try_distance(tol)
                    print('try again with new model an increase of the jitter')
                    break
                for it in range(n_cand_leaf):
                    index = it + t*n_cand_leaf
                    candidates[index, 0:dim] = candidate[it].numpy()
                    candidates[index, dim] = acq_value.numpy()

#            print('I (proc ', my_rank, ') propose candidates ', candidates)

            # 4. Master yields the candidates from workers
            crit = ExpectedImprovement(model._model, torch.max(model._train_Y), maximize = True)
            
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
#                print('going through leaf: ', subdomains._list[c]._index)
                get_from = c%n_proc
                if (get_from == 0):
                    for it in range(n_cand_leaf):
                        candidates_list.append(candidates[cpt, 0:dim])
                        acq_val = crit(torch.tensor(candidates[cpt,0:dim]).unsqueeze(0))
                        acq_value_list.append(acq_val.detach().numpy()[0]) # eval_f crit with AF model from master  
                        cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c) # n_cand_leaf candidates
                    for it in range(n_cand_leaf):
                        candidates_list.append(cand[it, 0:dim])                        
                        acq_val = crit(torch.tensor(cand[it,0:dim]).unsqueeze(0))
                        #print(acq_val)
                        acq_value_list.append(acq_val.detach().numpy()[0]) # eval_f crit with AF model from master
                    
                #print(acq_value_list)
                #print(type(acq_value_list[c*n_cand_leaf:(c+1)*n_cand_leaf]))
                subdomains._list[c]._crit = np.max(acq_value_list[c*n_cand_leaf:(c+1)*n_cand_leaf])
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
              
            print('Choosing ', batch_size, ' candidates among ', len(sort_X))
#            print(sort_X)
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
#            print(selected_candidates)
            t_ap = time()
            time_per_cycle[iter, 1] = time() - t_ap0
            time_per_cycle[iter, 2] = t_ap - t_start
        
            ######################################################################
            
            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])
                x_new = selected_candidates[n_proc*c] 
#                print(torch.tensor(x_new).unsqueeze(0))
                DB.add(torch.tensor(x_new, dtype = torch.float64).unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(torch.tensor(selected_candidates[c]).unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            T.update(subdomains)

            target[0, iter+1] = torch.min(DB._y).numpy()
            
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            print("Alg. g1_BSP-qEGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        print('main return')
        return target, time_per_cycle
    
    else: # workers
        init_size = DB._size      
        
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
#            print('I (proc ', my_rank, ') am in cycle ', iter)
            if(keep_going.sum() == 1):    
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                
                #print('I (proc ', my_rank, ') received the DB')
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
                
                DB.set_Xy(tmp_X, tmp_y)

                # 1. Workers receive the number of leaves to explore ; i.e number of AP to perform
                n_tasks = np.zeros(n_proc, dtype = 'i')
                comm.Bcast(n_tasks, root = 0)
                #print('I (proc ', my_rank, ') am receiving ', n_tasks, ' tasks')
    
                # 2. Workers receive the bounds to perform APs   
                my_tasks = n_tasks[my_rank] 
                b_list = []
                bounds = np.zeros((2, DB._dim))
                for t in range(my_tasks):
                    bounds = comm.recv(source = 0, tag = my_rank + t * n_proc)
                    b_list.append(bounds)
                #print('I need to perform an AP in each subdomain of the following list \n', b_list)
                
                # Create global model
                scaled_y = DB.min_max_y_scaling()
    #                scaled_y = DB_temp.normal_y_scaling()            
                model = GP_model(DB._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    crit = qExpectedImprovement(model._model, torch.max(model._train_Y))

                # 3. Perform AP for each subdomain
                candidates = np.zeros((n_cand_leaf*my_tasks, DB._dim + 1))
                for t in range(my_tasks):
                    bounds = b_list[t]
                    try :
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=n_cand_leaf, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        break
                    for it in range(n_cand_leaf):
                        index = it + t*n_cand_leaf
                        candidates[index, 0:dim] = candidate[it].numpy()
                        candidates[index, dim] = acq_value.numpy()
                
               # print('I (proc ', my_rank, ') propose candidates ', candidates)

                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    i0 = 0 + t*n_cand_leaf
                    i1 = n_cand_leaf + t*n_cand_leaf
                    comm.send(candidates[i0:i1,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
    
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

                cand = []
                for c in range(n_cand[my_rank]):
                    temp_cand = comm.recv(source = 0, tag = my_rank + c * n_proc)
                    cand.append(temp_cand)
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
            
            ## Receive from master 
            else :
                print('No more candidate, break')
                break
        print('return')
        return None        

#%% # No limit for number of leaves, but not all leaves are activated
def par_g1_BSP2_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, id_run, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    dim = DB._dim
    tol = 0.01# tolerance/min distance
    n_jobs = 2*n_proc # number of tasks per proc

    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)

        subdomains = T.get_leaves() # Returns an element of class Subdomains

        T.check_volume()
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_current = 0.
        t_0 = time()
        while (iter < n_cycle and t_current < t_max):
#        for iter in range(n_cycle):
            keep_going = np.ones(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

            t_start = time()
            # Define and fit model
            #print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.ones(n_proc, dtype = 'i')*2
            comm.Bcast(n_tasks, root = 0)

            # create probability vector
            crit_tot = 0.
            proba = np.zeros(n_leaves)
            for c in range(n_leaves):
                crit = np.abs(subdomains._list[c]._crit)
                proba[c] = crit
                crit_tot += crit
            proba = proba/crit_tot
#            print(proba)
                        
            # select subdomains according to probability
            indexes = np.random.choice(a=np.arange(0, n_leaves), size=n_jobs, replace=False, p=proba)
            activated_sbdms = Subdomains()
            for c in indexes:
#                print('Leaf ', subdomains._list[c]._index, ' is activated.')
                activated_sbdms._list.append(subdomains._list[c])
            assert activated_sbdms.get_size() == n_jobs, 'Check number of chosen sub-domains'
#            print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)

            # 2. Master sends the bounds to workers - Dispatch tasks (APs)
            my_tasks = n_tasks[my_rank]
            b_list = []
            for c in range(n_jobs):
                send_to = c%n_proc
                if send_to == 0:
                    bounds = activated_sbdms._list[c]._domain
                    b_list.append(bounds)
                else:
                    bounds = activated_sbdms._list[c]._domain
                    comm.send(bounds, dest = send_to, tag = c)

            # Create global model
            scaled_y = DB.min_max_y_scaling()
#                scaled_y = DB_temp.normal_y_scaling()            
            t_mod0 = time()
            model = GP_model(DB._X, scaled_y)
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                #model.fit()
                model.custom_fit(Global_Var.large_fit)  
                time_per_cycle[iter, 0] = time() - t_mod0
                crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)

            t_ap0 = time()
            
            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            for t in range(my_tasks):
                bounds = b_list[t]
                try :
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                except :
                    print('\n Failed to optimize the acquisition function. Why ?')
                    #DB_temp.try_distance(tol)
                    print('try again with new model an increase of the jitter')
                    break

                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.detach().numpy().reshape(1)
 #           print('I (proc ', my_rank, ') propose candidates ', candidates)

            # 4. Master yields the candidates from workers
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_jobs):
                get_from = c%n_proc
                if (get_from == 0):
                    candidates_list.append(candidates[cpt, 0:dim])
                    acq_value_list.append(candidates[cpt, dim])
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)
                    candidates_list.append(cand[0:dim])
                    tmp_crit = crit(torch.tensor(cand[0:dim]).unsqueeze(0))
                    acq_value_list.append(tmp_crit.detach().numpy()[0]) # eval_f crit with AF model from master
                activated_sbdms._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
              
   #         print('Choosing ', batch_size, ' candidates among ', len(sort_X))
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            t_ap = time()
            time_per_cycle[iter, 1] = time() - t_ap0
            time_per_cycle[iter, 2] = t_ap - t_start
        
            ######################################################################
            
            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])
                x_new = selected_candidates[n_proc*c] 
#                print(torch.tensor(x_new).unsqueeze(0))
                DB.add(torch.tensor(x_new, dtype = torch.float64).unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(torch.tensor(selected_candidates[c]).unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            T.update_twice(subdomains)
            target[0, iter+1] = torch.min(DB._y).numpy()
            
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            print("Alg. g1_BSP2-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        print('Tree is updated according to crit value. But crit values are not updated for every leaves since only a subset is activated')
        return target, time_per_cycle
    
    else: # workers
        init_size = DB._size      
        
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            if(keep_going.sum() == 1):    
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                
                #print('I (proc ', my_rank, ') received the DB')
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
                
                DB.set_Xy(tmp_X, tmp_y)

                # 1. Workers receive the number of leaves to explore ; i.e number of AP to perform
                n_tasks = np.zeros(n_proc, dtype = 'i')
                comm.Bcast(n_tasks, root = 0)
                #print('I (proc ', my_rank, ') am receiving ', n_tasks, ' tasks')
    
                # 2. Workers receive the bounds to perform APs   
                my_tasks = n_tasks[my_rank] 
                b_list = []
                bounds = np.zeros((2, DB._dim))
                for t in range(my_tasks):
                    bounds = comm.recv(source = 0, tag = my_rank + t * n_proc)
                    b_list.append(bounds)
#                print('I need to perform an AP in each subdomain of the following list \n', b_list)
                
                # Create global model
                scaled_y = DB.min_max_y_scaling()
    #                scaled_y = DB_temp.normal_y_scaling()            
                model = GP_model(DB._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)

                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB._dim + 1))

                for t in range(my_tasks):
                    bounds = b_list[t]
                    try :
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        break

                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.detach().numpy()
                #print('I (proc ', my_rank, ') propose candidates ', candidates)

                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
    
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

                cand = []
                for c in range(n_cand[my_rank]):
                    temp_cand = comm.recv(source = 0, tag = my_rank + c * n_proc)
                    cand.append(temp_cand)
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
            
            ## Receive from master 
            else :
                print('No more candidate, break')
                break
            
        return None        
