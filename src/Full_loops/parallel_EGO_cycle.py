#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:22:58 2022

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
from botorch.acquisition import UpperConfidenceBound
from torch.quasirandom import SobolEngine
from botorch.generation import MaxPosteriorSampling

from time import time

from botorch.optim import optimize_acqf
dtype = torch.double

#%% KB-qEGO algorithm
def par_KB_qEGO_run(DB, n_cycle, t_max, batch_size, id_run, crit_name = 'ei', comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()

    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))
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
                model.custom_fit(Global_Var.large_fit)
                #model.fit()
                #print("Model learning took --- %s seconds ---" % (t_model - t_start))
                t_model = time ()
                t_mod = t_model - t_start
    
                ######################################################################
                # Acquisition process
                bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
                DB_tmp = DB.copy()
                candidates = torch.zeros((batch_size, DB._dim))
                t_ap_sum = 0
                for k in range(batch_size):
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        if k!=0:
                            t_mod0 = time()
                            scaled_y = DB_tmp.min_max_y_scaling()
                            model = GP_model(DB_tmp._X, scaled_y)
                            model.custom_fit(Global_Var.small_fit) #much more stable
                            t_mod += (time()-t_mod0)
                            #model.fit()
                        t_ap0 = time()
                        if (crit_name == 'ei'):
                            crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
                        elif (crit_name == 'lcb'):
                            crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                        else:
                            raise NameError(crit_name + 'is an invalid name for KB qEGO criterion : use lcb or ei')
    
                        with gpytorch.settings.cholesky_jitter(Global_Var.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                        t_ap_sum += (time()-t_ap0)
                        #print('Candidate number ', k, ' is ', candidate)
                        DB_tmp.add(candidate, model.predict(candidate).clone().detach())
                        candidates[k,] = candidate.clone().detach()                
                del DB_tmp
                time_per_cycle[iter, 0] = t_mod
                t_ap = time()
                time_per_cycle[iter, 1] = t_ap_sum
                time_per_cycle[iter, 2] = t_ap - t_start 

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
                send_cand = candidates[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(candidates[n_proc*c].numpy())
                DB.add(candidates[n_proc*c].unsqueeze(0), torch.tensor(y_new))
                
            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(candidates[c].unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
            DB.try_distance()
            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start

            print("Alg. KB-qEGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())

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

        print('Final KB-$q$EGO value :', end='')
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
        
#%% Multi-criteria KB-qEGO
def par_MIC_qEGO_run(DB, n_cycle, t_max, batch_size, id_run, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()

    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
        #for iter in range(n_cycle):
            t_start = time()
            # Define and fit model
            # Output data scaled in [0, 1] - min_max_scaling
            scaled_y = DB.min_max_y_scaling()
            model = GP_model(DB._X, scaled_y)
            
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                model.custom_fit(Global_Var.large_fit)
               # model.fit()
                #print("Model learning took --- %s seconds ---" % (t_model - t_start))
                t_model = time ()
                t_mod = t_model - t_start
    
                ######################################################################
                # Acquisition process
                bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
                DB_tmp = DB.copy()
                candidates = torch.zeros((batch_size, DB._dim))
                k = 0
                t_ap_sum = 0
                while k < batch_size:
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        if k!=0:
                            t_mod0 = time()
                            scaled_y = DB_tmp.min_max_y_scaling()
                            model = GP_model(DB_tmp._X, scaled_y)
                            model.custom_fit(Global_Var.small_fit)
                            t_mod += (time() - t_mod0)
                            #model.fit()
                                
                        t_ap0 = time()
                        crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
                        crit2 = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    
                        with gpytorch.settings.cholesky_jitter(Global_Var.chol_jitter):
                            candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                        #print('Candidate number ', k, ' is ', candidate)
                        DB_tmp.add(candidate, model.predict(candidate).clone().detach())
                        candidates[k,] = candidate.clone().detach()
                        k = k + 1        
        
                        if(k + 1 < batch_size):
                            with gpytorch.settings.cholesky_jitter(Global_Var.chol_jitter):
                                candidate, acq_value = optimize_acqf(crit2, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                            #print('Candidate number ', k, ' is ', candidate)
                            DB_tmp.add(candidate, model.predict(candidate).clone().detach())
                            candidates[k,] = candidate.clone().detach()
                            k = k + 1
                        t_ap_sum += (time()-t_ap0)

                del DB_tmp
                
                time_per_cycle[iter, 0] = t_mod # t_models (sum)
                t_ap = time()
                time_per_cycle[iter, 1] = t_ap_sum # t_ap (sum)
                time_per_cycle[iter, 2] = t_ap - t_start
    
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
                send_cand = candidates[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)
            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(candidates[n_proc*c].numpy())
                DB.add(candidates[n_proc*c].unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(candidates[c].unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        

            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap # t_eval
            time_per_cycle[iter, 4] = t_end - t_start # t_cycle

            print("Alg. MIC_qEGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
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
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)

        #print(DB._X, DB._y)
        print('Final MIC_qEGO value :', end='')
        DB.print_min()
        return target, time_per_cycle

    else:
        for iter in range(n_cycle+1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute. ')

            comm.Bcast(n_cand, root = 0)
            #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

            if (n_cand.sum() == 0):
                break
            else :
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
        
#%% Classical qEGO with multipoint EI
def par_qEGO_run(DB, n_cycle, t_max, batch_size, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()

    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
#        for iter in range(n_cycle):
            t_start = time()
            Y_neg = -DB._y.clone().detach()

            # Define and fit model
            # Output data scaled in [0, 1] - min_max_scaling
            scaled_y = (Y_neg - Y_neg.min()) / (Y_neg.max()-Y_neg.min())
            model = GP_model(DB._X, scaled_y)

            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                model.custom_fit(Global_Var.large_fit)
                #model.fit()
                t_model = time ()
                time_per_cycle[iter, 0] = t_model - t_start
               #print("Model learning took --- %s seconds ---" % (t_model - t_start))
            
                ######################################################################
                # Acquisition process
                bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])    
                crit = qExpectedImprovement(model._model, torch.max(model._train_Y))
                with gpytorch.settings.cholesky_jitter(Global_Var.chol_jitter):
                    candidates, acq_value = optimize_acqf(crit, bounds=bounds, q=batch_size, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
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
                send_cand = candidates[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(candidates[n_proc*c].numpy())
                DB.add(candidates[n_proc*c].unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(candidates[c].unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        

            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 2] = t_ap - t_start 
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            #print(time_per_cycle[iter,:])
            print("Alg. Monte Carlo qEGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
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

#        print(DB._X, DB._y)
        print('Final qEGO value :', end='')
        DB.print_min()
        
        
        return target, time_per_cycle
    else:
        for iter in range(n_cycle+1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute at iter ', iter, '. ')

            comm.Bcast(n_cand, root = 0)
            #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

            if (n_cand.sum() == 0):
                break
            else :
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

#%% 
def par_Lanczos_qEGO_run(DB, n_cycle, t_max, batch_size, size_Lanczos, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()

    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
#        for iter in range(n_cycle):
            t_start = time()
            Y_neg = -DB._y.clone().detach()

            # Define and fit model
            # Output data scaled in [0, 1] - min_max_scaling
            scaled_y = (Y_neg - Y_neg.min()) / (Y_neg.max()-Y_neg.min())
            model = GP_model(DB._X, scaled_y)

            model.custom_fit(Global_Var.large_fit)
            #model.fit()
            t_model = time ()
            time_per_cycle[iter, 0] = t_model - t_start
            #print("Model learning took --- %s seconds ---" % (t_model - t_start))
         
            ######################################################################
            # Acquisition process
            bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])    
            crit = qExpectedImprovement(model._model, torch.max(model._train_Y))
            with gpytorch.settings.max_cholesky_size(size_Lanczos), gpytorch.settings.fast_computations(True), gpytorch.settings.fast_pred_samples(True), gpytorch.settings.fast_pred_var(True), gpytorch.settings.cholesky_jitter(Global_Var.chol_jitter):
                candidates, acq_value = optimize_acqf(crit, bounds=bounds, q=batch_size, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
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
                send_cand = candidates[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(candidates[n_proc*c].numpy())
                DB.add(candidates[n_proc*c].unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(candidates[c].unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        

            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 2] = t_ap - t_start 
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            print("Alg. Lanczos qEGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
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

#        print(DB._X, DB._y)
        print('Final Lanczos qEGO value :', end='')
        DB.print_min()
        
        
        return target, time_per_cycle
    else:
        for iter in range(n_cycle+1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute at iter ', iter, '. ')

            comm.Bcast(n_cand, root = 0)
            #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

            if (n_cand.sum() == 0):
                break
            else :
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

#%% qEGO Thompson Sampling
def par_TS_qEGO_run(DB, n_cycle, t_max, batch_size, id_run, comm):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    n_cand_sobol = 5000
    ### Main
    if my_rank == 0:
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
#        for iter in range(n_cycle):
            t_start = time()
            Y_neg = -DB._y.clone().detach()

            # Define and fit model
            # Output data scaled in [0, 1] - min_max_scaling
            scaled_y = (Y_neg - Y_neg.min()) / (Y_neg.max()-Y_neg.min())
            model = GP_model(DB._X, scaled_y)

            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                model.custom_fit(Global_Var.large_fit)
                #model.fit()
                t_model = time ()
                time_per_cycle[iter, 0] = t_model - t_start
               #print("Model learning took --- %s seconds ---" % (t_model - t_start))
            
                ######################################################################
                # Acquisition process
                bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])    
                sobol = SobolEngine(DB._X.shape[-1], scramble=True)
                X_cand = sobol.draw(n_cand_sobol).to(dtype=dtype)
                with torch.no_grad():
                    thompson_sampling = MaxPosteriorSampling(model=model._model, replacement=False)
                    candidates = thompson_sampling(X_cand, num_samples=batch_size)
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
                send_cand = candidates[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
                    #print('I m proc', my_rank, ' and I send ', send_cand, ' to proc ', send_to)

            ## eval_f
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(candidates[n_proc*c].numpy())
                DB.add(candidates[n_proc*c].unsqueeze(0), torch.tensor(y_new))

            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(candidates[c].unsqueeze(0), torch.tensor(recv_eval))

                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        

            target[0, iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 2] = t_ap - t_start 
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
#            print(time_per_cycle[iter,:])
            print("Alg. TS-qEGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
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

#        print(DB._X, DB._y)
        print('Final TS-qEGO value :', end='')
        DB.print_min()
        
        
        return target, time_per_cycle
    else:
        for iter in range(n_cycle+1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            #print('I am worker ', my_rank, 'and I wait for the number of candidates I have to compute at iter ', iter, '. ')

            comm.Bcast(n_cand, root = 0)
            #print('I am worker ', my_rank, 'and I have ', n_cand[my_rank], ' to compute')

            if (n_cand.sum() == 0):
                break
            else :
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


