#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:55:57 2023

@author: maxime

run with:
    mpiexec -n 4 python3 SDA_lBSP_EGO.py
"""
from mpi4py import MPI
import sys
sys.path.append('../src')
from time import time
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_proc = comm.Get_size()
print('From ', my_rank, ' : Running main with ', n_proc, 'proc')

from random import random
import numpy as np
import torch
import gpytorch

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.settings import cholesky_jitter

from Surrogates.GPyTorch_models import GP_model
from DataSets.DataSet import DataBase
import Global_Var
from Global_Var import *

from Problems.Ackley import Ackley
 
from BSP_tree.tree import Tree
from BSP_tree.split_functions import default_split
   

def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    print('From ', my_rank, ' : Running main with ', n_proc, 'proc')
     
    # Budget parameters
    dim = 6;
    batch_size = n_proc;
    budget = 16;
    t_max = 60;
    sim_cost = 5 # -1 if real cost
    print('The budget for this run is:', budget, ' cycles.')
    n_init = 96
    n_cycle = budget
    n_leaves = 2*n_proc
    tree_depth = int(np.log(n_leaves)/np.log(2))
    n_learn = min(n_init, 128)
   
    # Define the problem
    f = Ackley(dim)
    tol = 0.01 # tolerance/min distance



    
    if (my_rank == 0):
        DB = DataBase(f, n_init)
        r = random()*1000
        DB.par_create(comm = comm, seed = r)
            
        target = np.zeros(n_cycle+1)
        target[0] = torch.min(DB._y).numpy()
        time_per_cycle = np.zeros((n_cycle, 5))


        bounds = torch.stack([torch.ones(DB._dim)*(0), torch.ones(DB._dim)*1])
        T = Tree(DB, bounds)
        T.build(depth = tree_depth, split_function = default_split)
        
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            keep_going = np.ones(1, dtype = 'i')
            # Synchronize with all worker at the beginning of each cycle
            comm.Bcast(keep_going, root = 0)

            t_start = time()
            comm.Bcast(DB._X.numpy(), root = 0)
            comm.Bcast(DB._y.numpy(), root = 0)

            subdomains = T.get_leaves() # Returns an element of class Subdomains
            n_leaves = subdomains.get_size()
            T.check_volume()

            # 1. Master computes number of tasks for each process and broadcast it
            n_tasks = np.zeros(n_proc, dtype = 'i')
            for c in range(subdomains.get_size()):
                send_to = c%n_proc
                n_tasks[send_to] += 1
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

            # 3. Perform AP for each subdomain
            candidates = np.zeros((my_tasks, DB._dim + 1))
            t_mod = 0
            t_ap_sum = 0
            for t in range(my_tasks):
                bounds = b_list[t]
                center = (bounds[0]+bounds[1])*0.5
                DB_temp = DB.copy_from_center(n_learn, center)
                scaled_y = DB_temp.min_max_y_scaling()
                
                model = GP_model(DB_temp._X, scaled_y)
                with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                    t_mod0 = time()
                    model.custom_fit(Global_Var.large_fit)  
                    t_mod += time() - t_mod0
                    
                    # Use of UCB to be able to compare the results from different models
                    crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                    t_ap0 = time()
                    candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    t_ap_sum += time() - t_ap0
                                
                candidates[t, 0:dim] = candidate.numpy()
                candidates[t, dim] = acq_value.numpy()
                
                del model
                del DB_temp
            time_per_cycle[iter, 0] = t_mod
            time_per_cycle[iter, 1] = t_ap_sum

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
                candidates_list.append(cand[0:dim])
                acq_value_list.append(-cand[dim])
                subdomains._list[c]._crit = acq_value_list[c]
            
            # Sort the candidates to choose which ones are being evaluated
            try :
                sort_X = [el for _, el in sorted(zip(acq_value_list, candidates_list), reverse = True)]
            except :
                print('Failed to sort list')
                sort_X = candidates_list
            # Select clean also tests the distance between all the data
            selected_candidates = DB.select_clean(sort_X, batch_size, n_leaves, tol)
            t_ap = time()
            time_per_cycle[iter, 2] = t_ap - t_start

            # 5. Master sends candidates for evaluation
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(selected_candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1

            comm.Bcast(n_cand, root = 0)
            for c in range(len(selected_candidates)):
                send_cand = selected_candidates[c]
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = 1000 + c)
  
            ## Evaluate
            for c in range(int(n_cand[0])):
                y_new = DB.eval_f(selected_candidates[n_proc*c])
                DB.add(torch.tensor(selected_candidates[n_proc*c]).reshape(1,dim), torch.tensor(y_new))

            ## Gather
            for c in range(len(selected_candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = 1000 + c)
                    DB.add(torch.tensor(selected_candidates[c]).reshape(1,dim), torch.tensor(recv_eval))
        
            target[iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 3] = t_end - t_ap # t_eval
            time_per_cycle[iter, 4] = t_end - t_start
            arg_min = torch.argmin(DB._y)

            # Update the tree
            T.update(subdomains)
            
            print("Alg. lBSP-EGO, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ',  DB._y[arg_min])
            #DB.print_min()
            if(sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + sim_cost
            # print('t_current: ', t_current)
            # print('real time is ', time() - t_0)
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)

        DB.print_min()
        del T
        del DB
    else:
        DB_worker = DataBase(f, n_init) # in order to access function evaluation
        DB_worker.par_create(comm = comm)
    
        init_size = DB_worker._size
        for iter in range(n_cycle+1):
            keep_going = np.zeros(1, dtype = 'i')
            comm.Bcast(keep_going, root = 0)
            
            if(keep_going.sum() == 1):
                tmp_X = np.ndarray((init_size + iter * batch_size, dim))
                tmp_y = np.ndarray((init_size + iter * batch_size, 1))
                comm.Bcast(tmp_X, root = 0)
                comm.Bcast(tmp_y, root = 0)
    
                DB_worker.set_Xy(tmp_X, tmp_y)
                
                # 1. Workers receive the number of leaves to explore ; i.e number of AP to perform
                n_tasks = np.zeros(n_proc, dtype = 'i')
                comm.Bcast(n_tasks, root = 0)
    
                # 2. Workers receive the bounds to perform APs
                my_tasks = n_tasks[my_rank] 
                b_list = []
                bounds = np.zeros((2, DB_worker._dim))
                for t in range(my_tasks):
                    bounds = comm.recv(source = 0, tag = my_rank + t * n_proc)
                    b_list.append(bounds)
                
                # 3. Perform AP for each subdomain
                candidates = np.zeros((my_tasks, DB_worker._dim + 1))
                for t in range(my_tasks):
                    b_temp = b_list[t]
                    center = (b_temp[0]+b_temp[1])*0.5
                    DB_temp = DB_worker.copy_from_center(n_learn, center)
                    scaled_y = DB_temp.min_max_y_scaling()
                    if(torch.isnan(scaled_y).any() == True):
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp X: \n', DB_temp._X)
                        print(my_rank, ' iter ', iter, ' task ', t, ' DB temp y: \n', DB_temp._y)
                        print(my_rank, ' iter ', iter, ' task ', t, ' scaled train Y: \n', scaled_y)
    
                    model = GP_model(DB_temp._X, scaled_y)
                    with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                        model.custom_fit(Global_Var.large_fit)  
                        crit = UpperConfidenceBound(model._model, beta=0.1, maximize = False)
                        candidate, acq_value = optimize_acqf(crit, bounds=bounds, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)
                    candidates[t, 0:dim] = candidate.numpy()
                    candidates[t, dim] = acq_value.numpy()
                    
                    del model
                    del DB_temp

                # 4. Send back the found candidates to master
                for t in range(my_tasks):
                    comm.send(candidates[t,], dest = 0, tag = my_rank + t * n_proc)
    
                # 5. Get number of candidates to compute
                n_cand = np.zeros(n_proc, dtype = 'i')
                comm.Bcast(n_cand, root = 0)
            
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = 1000 + my_rank + c * n_proc))
    
                ## Evaluate
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB_worker.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = 1000 + my_rank + c * n_proc)
            else : 
                break
        del DB_worker
    
if __name__ == "__main__":
    main()
