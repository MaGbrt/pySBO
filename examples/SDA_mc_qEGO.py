#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:15:48 2023

@author: maxime

run with:
    mpiexec -n 4 python3 run_BSP_EGO.py
"""
from mpi4py import MPI
import sys
sys.path.append('../src')
from time import time

from random import random
import numpy as np
import torch
import gpytorch
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from Surrogates.GPyTorch_models import GP_model
from Problems.Composition_AS import Composition_AS
from DataSets.DataSet import DataBase
import Global_Var
from Global_Var import *


def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    print('From ', my_rank, ' : Running main with ', n_proc, 'proc')
     
    # Budget parameters
    dim = 10;
    batch_size = n_proc;
    budget = 160;
    t_max = 600;
    sim_cost = 5 # -1 if real cost
    print('The budget for this run is:', budget, ' cycles.')
    n_init = 96
    n_cycle = budget
       
    # Define the problem
    f = Composition_AS(dim)
    
    # Multi-criteria qEGO
    if my_rank == 0:
        DB = DataBase(f, n_init) # DB contains mapped individuals (into [0,1]^n_dvar) of torch type
        r = random()*1000
        DB.par_create(comm = comm, seed = r)

        target = np.zeros(n_cycle+1)
        target[0] = torch.min(DB._y).numpy()
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
            ## Evaluate
            for c in range(int(n_cand[0])):
                #y_new = DB._obj.evaluate(candidates[n_proc*c].numpy())
                y_new = DB.eval_f(candidates[n_proc*c].numpy()) # Eval_f also unmap the candidate
                DB.add(candidates[n_proc*c].unsqueeze(0), torch.tensor(y_new))
    
            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    DB.add(candidates[c].unsqueeze(0), torch.tensor(recv_eval))
    
                    #print('I m proc', my_rank, ' and I receive ', recv_eval, ' from proc ', get_from)
        
    
            target[iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap # t_eval
            time_per_cycle[iter, 4] = t_end - t_start # t_cycle
    
            print("Alg. MC_qEGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
            if(sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)
    
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)
    
        #print(DB._X, DB._y)
        print('Final MC_q-EGO value :', end='')
        DB.print_min()
        del DB

    else:
        DB_worker = DataBase(f, n_init) # in order to access function evaluation
        DB_worker.par_create(comm = comm)

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
    
                ## Evaluate
                y_new = []
                for c in range(n_cand[my_rank]):
#                    y_new.append(DB._obj.evaluate(cand[c]))
                    y_new.append(DB_worker.eval_f(cand[c])) # Eval_f also unmap the candidate
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
        del DB_worker

if __name__ == "__main__":
    main()
