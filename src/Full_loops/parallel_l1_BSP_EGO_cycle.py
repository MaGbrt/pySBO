#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:22:36 2022

@author: maxime
"""

import numpy as np
import torch
import gpytorch
import Global_Var
from Global_Var import *

from Surrogates.GPyTorch_models import GP_model
from Surrogates.SKIP_GP_models import SKIP_GP_model

from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.settings import cholesky_jitter

from BSP_tree.tree import Tree
from BSP_tree.split_functions import default_split
from time import time

dtype = torch.double

#%% Multiple local surrogate models that drive the optimization with Binary Space Partition acquisition process
def par_l1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, n_learn, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    tol = 0.01# tolerance/min distance

    dim = DB._dim
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = torch.min(DB._y).numpy()

        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        print('len of DB to send: ', len(DB._X))
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            keep_going = np.ones(1, dtype = 'i')
            # Synchronize with all worker at the beginning of each cycle
            comm.Bcast(keep_going, root = 0)

#        for iter in range(n_cycle):
            t_start = time()
            print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
#            print('Tree is built with ', n_leaves, ' leaves.')
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(subdomains.get_size()):
                send_to = c%n_proc
                n_tasks[send_to] += 1
#            print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)
            # Sending the number of tasks to compute to all workers
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

#            print('DB X and y', DB._X, DB._y)
            t_model = time ()
            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            t_mod = 0
            t_ap_sum = 0
            for t in range(my_tasks):
                bounds = b_list[t]
                center = (bounds[0]+bounds[1])*0.5
                DB_temp = DB.copy_from_center(n_learn, center)
                scaled_y = DB_temp.min_max_y_scaling()
#                scaled_y = DB_temp.normal_y_scaling()

                if(torch.isnan(scaled_y).any() == True):
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                    print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                
                model = GP_model(DB_temp._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    t_mod0 = time()
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    t_mod += time() - t_mod0
                    
                    # Use of UCB to be able to compare the results from different models
                    crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    #crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
 #                   print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                    #DB_temp.try_distance(tol)
                    t_ap0 = time()
                    try :
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        model = GP_model(DB_temp._X, scaled_y)
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            #model.fit()
                            model.custom_fit(Global_Var.large_fit)  

                        with cholesky_jitter(Global_Var.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    t_ap_sum += time() - t_ap0
                                
                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
                
                del model
                del DB_temp
            time_per_cycle[iter, 0] = t_mod
            time_per_cycle[iter, 1] = t_ap_sum

            #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')

            # 4. Master yields the candidates from workers
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
                get_from = c%n_proc
                if (get_from == 0):
                    cand = candidates[cpt,:]
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)

#                print('Get cand ', cand)
                candidates_list.append(cand[0:dim])
#                print('Setting -crit if LCB ', -cand[dim], ' leaf ', subdomains._list[c]._index)
                acq_value_list.append(-cand[dim])
                subdomains._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
            # print('AF values\n', acq_value_list)
            # print('Selected candidates\n', sort_X)
  #          print('Choosing ', batch_size, ' candidates among ', len(sort_X))
            # Select clean also tests the distance between all the data
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            # for cand in selected_candidates:
            #     print('Selected candidate\n', cand)
            t_ap = time()
            time_per_cycle[iter, 2] = t_ap - t_start

            # 5. Master sends candidates for evaluation
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = 1000 + c)
   #                 print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            #print(selected_candidates)
            #print(type(selected_candidates))
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])

                DB.add(torch.tensor(selected_candidates[n_proc*c]).reshape(1,dim), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = 1000 + c)
                    DB.add(torch.tensor(selected_candidates[c]).reshape(1,dim), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap # t_eval
            time_per_cycle[iter, 4] = t_end - t_start
            arg_min = torch.argmin(DB._y)

            # Update the tree
            T.update(subdomains)
            
            print("Alg. lBSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ',  DB._y[arg_min])
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            #t_current = time() - t_0
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        del T
        return target, time_per_cycle
                    
    else:
        init_size = DB._size
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            
            if(keep_going.sum() == 1):
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                print('len of tmp_X: ', len(tmp_X))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
    
                print('I (proc ', my_rank, ') received the DB')
                DB.set_Xy(tmp_X, tmp_y)
                #print('DB_X from ', my_rank, '\n', DB._X)
                #print('DB_y from ', my_rank, '\n', DB._y)
                
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
                
                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB._dim + 1))
     #           print('DB X and y', DB._X, DB._y)
                for t in range(my_tasks):
                    b_temp = b_list[t]
                    center = (b_temp[0]+b_temp[1])*0.5
                    DB_temp = DB.copy_from_center(n_learn, center)
                    scaled_y = DB_temp.min_max_y_scaling()
                    #scaled_y = DB_temp.normal_y_scaling()
                    if(torch.isnan(scaled_y).any() == True):
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                        print(my_rank, ' iter ', iter, ' task ', t, ' scaled train Y: \n', scaled_y)
    
                    model = GP_model(DB_temp._X, scaled_y)
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        model.custom_fit(Global_Var.large_fit)  
                        #model.fit()
                        crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
#                        crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
                        #DB_temp.try_distance(tol)
                        try :
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                        except :
                            print('\n Failed to optimize the acquisition function. Why ?')
                            #DB_temp.try_distance(tol)
                            print('try again with new model an increase of the jitter')
                            model = GP_model(DB_temp._X, scaled_y)
                            model.custom_fit(Global_Var.large_fit)  
                            #model.fit()

                            with cholesky_jitter(Global_Var.chol_jitter):
                                candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)

                            # candidate, acq_value = optimize_acqf(crit, bounds=b_temp, q=1, num_restarts=10, raw_samples=512)
                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                    
                    del model
                    del DB_temp
                #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')
                
                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')
            
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = 1000 + my_rank + c * n_proc))
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = 1000 + my_rank + c * n_proc)
            else : 
                break
                
        return None        

#%% Ajout d'une composante globale similaire a qEGO pour guider mieux l'evlotution de l'arbre
def par_lg1_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, n_learn, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    tol = 0.01# tolerance/min distance

    dim = DB._dim
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = torch.min(DB._y).numpy()

        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            keep_going = np.ones(1, dtype = 'i')
            
            t_gm = time()
            # Create a global model // Should be faster
            if iter % 5 == 0:
                g_scaled_y = DB.min_max_y_scaling()
                g_model = GP_model(DB._X, g_scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    g_model.custom_fit(Global_Var.medium_fit)  
                    g_crit = UpperConfidenceBound(g_model._model, beta=0.1, maximize = False)
            print('Time taken for the global model', time()-t_gm)
            # Synchronize with all worker at the beginning of each cycle
            comm.Bcast(keep_going, root = 0)

#        for iter in range(n_cycle):
            t_start = time()
#            print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
#            print('Tree is built with ', n_leaves, ' leaves.')
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(subdomains.get_size()):
                send_to = c%n_proc
                n_tasks[send_to] += 1
#            print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)
            # Sending the number of tasks to compute to all workers
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

#            print('DB X and y', DB._X, DB._y)
            t_model = time ()
            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            t_mod = 0
            t_ap_sum = 0
            for t in range(my_tasks):
                bounds = b_list[t]
                center = (bounds[0]+bounds[1])*0.5
                DB_temp = DB.copy_from_center(n_learn, center)
                scaled_y = DB_temp.min_max_y_scaling()
#                scaled_y = DB_temp.normal_y_scaling()

                if(torch.isnan(scaled_y).any() == True):
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                    print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                
                model = GP_model(DB_temp._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    t_mod0 = time()
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    t_mod += time() - t_mod0
                    
                    # Use of UCB to be able to compare the results from different models
                    crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    #crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
 #                   print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                    #DB_temp.try_distance(tol)
                    t_ap0 = time()
                    try :
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        model = GP_model(DB_temp._X, scaled_y)
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            #model.fit()
                            model.custom_fit(Global_Var.large_fit)  

                        with cholesky_jitter(Global_Var.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    t_ap_sum += time() - t_ap0
                                
                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
                
                del model
                del DB_temp
            time_per_cycle[iter, 0] = t_mod
            time_per_cycle[iter, 1] = t_ap_sum

            #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')

            # 4. Master yields the candidates from workers
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
                get_from = c%n_proc
                if (get_from == 0):
                    cand = candidates[cpt,:]
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)

#                print('Get cand ', cand)
                candidates_list.append(cand[0:dim])
#                print('Setting -crit if LCB ', -cand[dim], ' leaf ', subdomains._list[c]._index)
#                acq_value_list.append(-cand[dim])
                # print('Global crit evaluation (LCB), value must be minimized')
                # print('Evaluating:', torch.tensor(cand[0:dim]).unsqueeze(0))
                g_crit_res = -g_crit(torch.tensor(cand[0:dim]).unsqueeze(0))
                acq_value_list.append(g_crit_res.detach().numpy()[0]) # Reverse sign because sort function sorts decreasingly
                subdomains._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
            # print('AF values\n', acq_value_list)
            # print('Selected candidates\n', sort_X)
  #          print('Choosing ', batch_size, ' candidates among ', len(sort_X))
            # Select clean also tests the distance between all the data
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            # for cand in selected_candidates:
            #     print('Selected candidate\n', cand)
            t_ap = time()
            time_per_cycle[iter, 2] = t_ap - t_start

            # 5. Master sends candidates for evaluation
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = 1000 + c)
   #                 print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            #print(selected_candidates)
            #print(type(selected_candidates))
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])

                DB.add(torch.tensor(selected_candidates[n_proc*c]).reshape(1,dim), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = 1000 + c)
                    DB.add(torch.tensor(selected_candidates[c]).reshape(1,dim), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            # Update the tree
            T.update(subdomains)
            
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            arg_min = torch.argmin(DB._y)

            print("Alg. lg1BSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ',  DB._y[arg_min])
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            #t_current = time() - t_0
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        del T
        return target, time_per_cycle
                    
    else:
        init_size = DB._size
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            
            if(keep_going.sum() == 1):
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
    
                #print('I (proc ', my_rank, ') received the DB')
                DB.set_Xy(tmp_X, tmp_y)
                #print('DB_X from ', my_rank, '\n', DB._X)
                #print('DB_y from ', my_rank, '\n', DB._y)
                
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
                
                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB._dim + 1))
     #           print('DB X and y', DB._X, DB._y)
                for t in range(my_tasks):
                    b_temp = b_list[t]
                    center = (b_temp[0]+b_temp[1])*0.5
                    DB_temp = DB.copy_from_center(n_learn, center)
                    scaled_y = DB_temp.min_max_y_scaling()
                    #scaled_y = DB_temp.normal_y_scaling()
                    if(torch.isnan(scaled_y).any() == True):
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                        print(my_rank, ' iter ', iter, ' task ', t, ' scaled train Y: \n', scaled_y)
    
                    model = GP_model(DB_temp._X, scaled_y)
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        model.custom_fit(Global_Var.large_fit)  
                        #model.fit()
                        crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
#                        crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
                        #DB_temp.try_distance(tol)
                        try :
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                        except :
                            print('\n Failed to optimize the acquisition function. Why ?')
                            #DB_temp.try_distance(tol)
                            print('try again with new model an increase of the jitter')
                            model = GP_model(DB_temp._X, scaled_y)
                            model.custom_fit(Global_Var.large_fit)  
                            #model.fit()

                            with cholesky_jitter(Global_Var.chol_jitter):
                                candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)

                            # candidate, acq_value = optimize_acqf(crit, bounds=b_temp, q=1, num_restarts=10, raw_samples=512)
                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                    
                    del model
                    del DB_temp
                #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')
                
                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')
            
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = 1000 + my_rank + c * n_proc))
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = 1000 + my_rank + c * n_proc)
            else : 
                break
                
        return None        

#%% Ajout d'une composante globale similaire a qEGO pour guider mieux l'evlotution de l'arbre, mais un modele global plus rapide
def par_lg2_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, n_learn, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    tol = 0.01# tolerance/min distance

    dim = DB._dim
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = torch.min(DB._y).numpy()

        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            keep_going = np.ones(1, dtype = 'i')
            
            t_gm = time()
            # Create a global model // Should be faster
            if iter % 2 == 0:    
                g_scaled_y = DB.min_max_y_scaling()
                g_model = SKIP_GP_model(DB._X, g_scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    g_model.fit(Global_Var.small_fit)  
                    g_crit = ExpectedImprovement(g_model._model, torch.min(g_model._train_Y), maximize = False)
                    #UpperConfidenceBound(g_model._model, beta=0.1, maximize = False)
            print('Time taken for the global model', time()-t_gm)
            # Synchronize with all worker at the beginning of each cycle
            comm.Bcast(keep_going, root = 0)

#        for iter in range(n_cycle):
            t_start = time()
#            print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
#            print('Tree is built with ', n_leaves, ' leaves.')
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(subdomains.get_size()):
                send_to = c%n_proc
                n_tasks[send_to] += 1
#            print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)
            # Sending the number of tasks to compute to all workers
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

#            print('DB X and y', DB._X, DB._y)
            t_model = time ()
            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            t_mod = 0
            t_ap_sum = 0
            for t in range(my_tasks):
                bounds = b_list[t]
                center = (bounds[0]+bounds[1])*0.5
                DB_temp = DB.copy_from_center(n_learn, center)
                scaled_y = DB_temp.min_max_y_scaling()
#                scaled_y = DB_temp.normal_y_scaling()

                if(torch.isnan(scaled_y).any() == True):
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                    print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                
                model = GP_model(DB_temp._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    t_mod0 = time()
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    t_mod += time() - t_mod0
                    
                    # Use of UCB to be able to compare the results from different models
                    crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    #crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
 #                   print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                    #DB_temp.try_distance(tol)
                    t_ap0 = time()
                    try :
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        model = GP_model(DB_temp._X, scaled_y)
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            #model.fit()
                            model.custom_fit(Global_Var.large_fit)  

                        with cholesky_jitter(Global_Var.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    t_ap_sum += time() - t_ap0
                                
                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
                
                del model
                del DB_temp
            time_per_cycle[iter, 0] = t_mod
            time_per_cycle[iter, 1] = t_ap_sum

            #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')

            # 4. Master yields the candidates from workers
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
                get_from = c%n_proc
                if (get_from == 0):
                    cand = candidates[cpt,:]
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)

#                print('Get cand ', cand)
                candidates_list.append(cand[0:dim])
#                print('Setting -crit if LCB ', -cand[dim], ' leaf ', subdomains._list[c]._index)
#                acq_value_list.append(-cand[dim])
                print('Global crit evaluation (LCB), value must be minimized')
                print('Evaluating:', torch.tensor(cand[0:dim]).unsqueeze(0))
                g_crit_res = -g_crit(torch.tensor(cand[0:dim]).unsqueeze(0))
                acq_value_list.append(g_crit_res.detach().numpy()[0]) # Reverse sign because sort function sorts decreasingly
                subdomains._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
            # print('AF values\n', acq_value_list)
            # print('Selected candidates\n', sort_X)
  #          print('Choosing ', batch_size, ' candidates among ', len(sort_X))
            # Select clean also tests the distance between all the data
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            # for cand in selected_candidates:
            #     print('Selected candidate\n', cand)
            t_ap = time()
            time_per_cycle[iter, 2] = t_ap - t_start

            # 5. Master sends candidates for evaluation
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = 1000 + c)
   #                 print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            #print(selected_candidates)
            #print(type(selected_candidates))
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])

                DB.add(torch.tensor(selected_candidates[n_proc*c]).reshape(1,dim), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = 1000 + c)
                    DB.add(torch.tensor(selected_candidates[c]).reshape(1,dim), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            # Update the tree
            T.update(subdomains)
            
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            arg_min = torch.argmin(DB._y)

            print("Alg. lg2_BSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ',  DB._y[arg_min])
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            #t_current = time() - t_0
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        del T
        return target, time_per_cycle
                    
    else:
        init_size = DB._size
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            
            if(keep_going.sum() == 1):
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
    
                #print('I (proc ', my_rank, ') received the DB')
                DB.set_Xy(tmp_X, tmp_y)
                #print('DB_X from ', my_rank, '\n', DB._X)
                #print('DB_y from ', my_rank, '\n', DB._y)
                
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
                
                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB._dim + 1))
     #           print('DB X and y', DB._X, DB._y)
                for t in range(my_tasks):
                    b_temp = b_list[t]
                    center = (b_temp[0]+b_temp[1])*0.5
                    DB_temp = DB.copy_from_center(n_learn, center)
                    scaled_y = DB_temp.min_max_y_scaling()
                    #scaled_y = DB_temp.normal_y_scaling()
                    if(torch.isnan(scaled_y).any() == True):
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                        print(my_rank, ' iter ', iter, ' task ', t, ' scaled train Y: \n', scaled_y)
    
                    model = GP_model(DB_temp._X, scaled_y)
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        model.custom_fit(Global_Var.large_fit)  
                        #model.fit()
                        crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
#                        crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
                        #DB_temp.try_distance(tol)
                        try :
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                        except :
                            print('\n Failed to optimize the acquisition function. Why ?')
                            #DB_temp.try_distance(tol)
                            print('try again with new model an increase of the jitter')
                            model = GP_model(DB_temp._X, scaled_y)
                            model.custom_fit(Global_Var.large_fit)  
                            #model.fit()

                            with cholesky_jitter(Global_Var.chol_jitter):
                                candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)

                            # candidate, acq_value = optimize_acqf(crit, bounds=b_temp, q=1, num_restarts=10, raw_samples=512)
                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                    
                    del model
                    del DB_temp
                #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')
                
                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')
            
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = 1000 + my_rank + c * n_proc))
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = 1000 + my_rank + c * n_proc)
            else : 
                break
                
        return None        

#%% Reduction de toutes les dimensions dans les recherches locales (domain*0.9)
def par_l11_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, n_learn, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    tol = 0.01# tolerance/min distance

    dim = DB._dim
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = torch.min(DB._y).numpy()

        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            keep_going = np.ones(1, dtype = 'i')
            # Synchronize with all worker at the beginning of each cycle
            comm.Bcast(keep_going, root = 0)

#        for iter in range(n_cycle):
            t_start = time()
#            print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
#            print('Tree is built with ', n_leaves, ' leaves.')
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(subdomains.get_size()):
                send_to = c%n_proc
                n_tasks[send_to] += 1
#            print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)
            # Sending the number of tasks to compute to all workers
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

#            print('DB X and y', DB._X, DB._y)
            t_model = time ()
            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            t_mod = 0
            t_ap_sum = 0
            for t in range(my_tasks):
                bounds = b_list[t]*(0.98**iter)
                center = (bounds[0]+bounds[1])*0.5
                DB_temp = DB.copy_from_center(n_learn, center)
                scaled_y = DB_temp.min_max_y_scaling()
#                scaled_y = DB_temp.normal_y_scaling()

                if(torch.isnan(scaled_y).any() == True):
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                    print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                
                model = GP_model(DB_temp._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    t_mod0 = time()
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    t_mod += time() - t_mod0
                    
                    # Use of UCB to be able to compare the results from different models
                    crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    #crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
 #                   print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                    #DB_temp.try_distance(tol)
                    t_ap0 = time()
                    try :
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        model = GP_model(DB_temp._X, scaled_y)
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            #model.fit()
                            model.custom_fit(Global_Var.large_fit)  

                        with cholesky_jitter(Global_Var.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    t_ap_sum += time() - t_ap0
                                
                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
                
                del model
                del DB_temp
            time_per_cycle[iter, 0] = t_mod
            time_per_cycle[iter, 1] = t_ap_sum

            #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')

            # 4. Master yields the candidates from workers
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
                get_from = c%n_proc
                if (get_from == 0):
                    cand = candidates[cpt,:]
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)

#                print('Get cand ', cand)
                candidates_list.append(cand[0:dim])
#                print('Setting -crit if LCB ', -cand[dim], ' leaf ', subdomains._list[c]._index)
                acq_value_list.append(-cand[dim])
                subdomains._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
            # print('AF values\n', acq_value_list)
            # print('Selected candidates\n', sort_X)
  #          print('Choosing ', batch_size, ' candidates among ', len(sort_X))
            # Select clean also tests the distance between all the data
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            # for cand in selected_candidates:
            #     print('Selected candidate\n', cand)
            t_ap = time()
            time_per_cycle[iter, 2] = t_ap - t_start

            # 5. Master sends candidates for evaluation
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = 1000 + c)
   #                 print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            #print(selected_candidates)
            #print(type(selected_candidates))
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])

                DB.add(torch.tensor(selected_candidates[n_proc*c]).reshape(1,dim), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = 1000 + c)
                    DB.add(torch.tensor(selected_candidates[c]).reshape(1,dim), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            # Update the tree
            T.update(subdomains)
            
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            arg_min = torch.argmin(DB._y)

            print("Alg. l2BSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ',  DB._y[arg_min])
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            #t_current = time() - t_0
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        del T
        return target, time_per_cycle
                    
    else:
        init_size = DB._size
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            
            if(keep_going.sum() == 1):
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
    
                #print('I (proc ', my_rank, ') received the DB')
                DB.set_Xy(tmp_X, tmp_y)
                #print('DB_X from ', my_rank, '\n', DB._X)
                #print('DB_y from ', my_rank, '\n', DB._y)
                
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
                
                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB._dim + 1))
     #           print('DB X and y', DB._X, DB._y)
                for t in range(my_tasks):
                    b_temp = b_list[t]*(0.98**iter)
                    center = (b_temp[0]+b_temp[1])*0.5
                    DB_temp = DB.copy_from_center(n_learn, center)
                    scaled_y = DB_temp.min_max_y_scaling()
                    #scaled_y = DB_temp.normal_y_scaling()
                    if(torch.isnan(scaled_y).any() == True):
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                        print(my_rank, ' iter ', iter, ' task ', t, ' scaled train Y: \n', scaled_y)
    
                    model = GP_model(DB_temp._X, scaled_y)
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        model.custom_fit(Global_Var.large_fit)  
                        #model.fit()
                        crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
#                        crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
                        #DB_temp.try_distance(tol)
                        try :
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                        except :
                            print('\n Failed to optimize the acquisition function. Why ?')
                            #DB_temp.try_distance(tol)
                            print('try again with new model an increase of the jitter')
                            model = GP_model(DB_temp._X, scaled_y)
                            model.custom_fit(Global_Var.large_fit)  
                            #model.fit()

                            with cholesky_jitter(Global_Var.chol_jitter):
                                candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)

                            # candidate, acq_value = optimize_acqf(crit, bounds=b_temp, q=1, num_restarts=10, raw_samples=512)
                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                    
                    del model
                    del DB_temp
                #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')
                
                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')
            
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = 1000 + my_rank + c * n_proc))
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = 1000 + my_rank + c * n_proc)
            else : 
                break
                
        return None        

#%% Trying to make the tree evolve faster - remove the constraint of covering all the search space
# Activate the subregion according to target value (probability based on y)
# IDEE NON IMPLEMENTEE : si pas d'amelioration pendant k iter, on coupe des feuilles profondes
def par_l2_BSP_EGO_run(DB, n_cycle, t_max, batch_size, tree_depth, n_learn, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    tol = 0.01# tolerance/min distance

    dim = DB._dim
    if my_rank == 0:
        n_init_leaves = pow(2, tree_depth)
        max_leaves = n_init_leaves
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = torch.min(DB._y).numpy()

        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)
        
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            keep_going = np.ones(1, dtype = 'i')
            # Synchronize with all worker at the beginning of each cycle
            comm.Bcast(keep_going, root = 0)

#        for iter in range(n_cycle):
            t_start = time()
#            print('Master broadcast DB to all')
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
            print('Tree is built with ', n_leaves, ' leaves.')
            subdomains.select(max_leaves)
            n_leaves = subdomains.get_size()
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(subdomains.get_size()):
                send_to = c%n_proc
                n_tasks[send_to] += 1
#            print('I am proc ', my_rank, 'and I broadcast n_tasks ', n_tasks)
            # Sending the number of tasks to compute to all workers
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

#            print('DB X and y', DB._X, DB._y)
            t_model = time ()
            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            t_mod = 0
            t_ap_sum = 0
            for t in range(my_tasks):
                bounds = b_list[t]
                center = (bounds[0]+bounds[1])*0.5
                DB_temp = DB.copy_from_center(n_learn, center)
                scaled_y = DB_temp.min_max_y_scaling()
#                scaled_y = DB_temp.normal_y_scaling()

                if(torch.isnan(scaled_y).any() == True):
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                    print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                    print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                
                model = GP_model(DB_temp._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    t_mod0 = time()
                    #model.fit()
                    model.custom_fit(Global_Var.large_fit)  
                    t_mod += time() - t_mod0
                    
                    # Use of UCB to be able to compare the results from different models
                    crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    #crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
 #                   print(my_rank, ' iter ', iter, ' task ', t,' scaled train Y: \n', scaled_y)
                    #DB_temp.try_distance(tol)
                    t_ap0 = time()
                    try :
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    except :
                        print('\n Failed to optimize the acquisition function. Why ?')
                        #DB_temp.try_distance(tol)
                        print('try again with new model an increase of the jitter')
                        model = GP_model(DB_temp._X, scaled_y)
                        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                            #model.fit()
                            model.custom_fit(Global_Var.large_fit)  

                        with cholesky_jitter(Global_Var.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    t_ap_sum += time() - t_ap0
                                
                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
                
                del model
                del DB_temp
            time_per_cycle[iter, 0] = t_mod
            time_per_cycle[iter, 1] = t_ap_sum

            #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')

            # 4. Master yields the candidates from workers
            candidates_list = []
            acq_value_list = []
            cpt = 0
            for c in range(n_leaves):
                get_from = c%n_proc
                if (get_from == 0):
                    cand = candidates[cpt,:]
                    cpt += 1
                else :
                    cand = comm.recv(source = get_from, tag = c)

#                print('Get cand ', cand)
                candidates_list.append(cand[0:dim])
#                print('Setting -crit if LCB ', -cand[dim], ' leaf ', subdomains._list[c]._index)
                acq_value_list.append(-cand[dim])
                subdomains._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being eval_fd
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
            # print('AF values\n', acq_value_list)
            # print('Selected candidates\n', sort_X)
  #          print('Choosing ', batch_size, ' candidates among ', len(sort_X))
            # Select clean also tests the distance between all the data
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            # for cand in selected_candidates:
            #     print('Selected candidate\n', cand)
            t_ap = time()
            time_per_cycle[iter, 2] = t_ap - t_start

            # 5. Master sends candidates for evaluation
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            #print('I am proc ', my_rank, 'and I broadcast n_cand ', n_cand)
            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = 1000 + c)
   #                 print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            #print(selected_candidates)
            #print(type(selected_candidates))
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])

                DB.add(torch.tensor(selected_candidates[n_proc*c]).reshape(1,dim), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = 1000 + c)
                    DB.add(torch.tensor(selected_candidates[c]).reshape(1,dim), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap # t_eval
            time_per_cycle[iter, 4] = t_end - t_start
            arg_min = torch.argmin(DB._y)

            # Update the tree
            T.update_split_only(subdomains)
            
            print("Alg. l2BSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ',  DB._y[arg_min])
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
            #t_current = time() - t_0
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        del T
        return target, time_per_cycle
                    
    else:
        init_size = DB._size
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            
            if(keep_going.sum() == 1):
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
    
                #print('I (proc ', my_rank, ') received the DB')
                DB.set_Xy(tmp_X, tmp_y)
                #print('DB_X from ', my_rank, '\n', DB._X)
                #print('DB_y from ', my_rank, '\n', DB._y)
                
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
                
                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB._dim + 1))
     #           print('DB X and y', DB._X, DB._y)
                for t in range(my_tasks):
                    b_temp = b_list[t]
                    center = (b_temp[0]+b_temp[1])*0.5
                    DB_temp = DB.copy_from_center(n_learn, center)
                    scaled_y = DB_temp.min_max_y_scaling()
                    #scaled_y = DB_temp.normal_y_scaling()
                    if(torch.isnan(scaled_y).any() == True):
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                        print(my_rank, ' iter ', iter, ' task ', t, ' scaled train Y: \n', scaled_y)
    
                    model = GP_model(DB_temp._X, scaled_y)
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        model.custom_fit(Global_Var.large_fit)  
                        #model.fit()
                        crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
#                        crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
                        #DB_temp.try_distance(tol)
                        try :
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                        except :
                            print('\n Failed to optimize the acquisition function. Why ?')
                            #DB_temp.try_distance(tol)
                            print('try again with new model an increase of the jitter')
                            model = GP_model(DB_temp._X, scaled_y)
                            model.custom_fit(Global_Var.large_fit)  
                            #model.fit()

                            with cholesky_jitter(Global_Var.chol_jitter):
                                candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)

                            # candidate, acq_value = optimize_acqf(crit, bounds=b_temp, q=1, num_restarts=10, raw_samples=512)
                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                    
                    del model
                    del DB_temp
                #print('I (proc ', my_rank, ') have been asked to perform ', my_tasks, ' tasks and I return ', candidates, ' to master process')
                
                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')
                comm.Bcast(n_cand, root = 0)
                #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')
            
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = 1000 + my_rank + c * n_proc))
    
                ## eval_f
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = 1000 + my_rank + c * n_proc)
            else : 
                break
                
        return None        
