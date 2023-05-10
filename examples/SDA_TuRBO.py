#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:15:48 2023

@author: maxime

run with:
    mpiexec -n 4 python3 SDA_TuRBO.py
"""
from mpi4py import MPI
import sys
sys.path.append('../src')
from time import time

from random import random
import numpy as np
import torch
import gpytorch
from Surrogates.GPyTorch_models import GP_model
from Problems.Composition_AS import Composition_AS
from DataSets.DataSet import DataBase
from TuRBO.TuRBO_class import TuRBO

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
    budget = 16;
    t_max = 60;
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
        Turbo = TuRBO(DB._dim, batch_size)
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        
        iter = 0
        t_0 = time()
        t_current = 0.
        while (iter < n_cycle and t_current < t_max):
            t_start = time()
            Y_turbo = -DB._y.clone().detach()
            scaled_y = (Y_turbo - Y_turbo.min()) / (Y_turbo.max()-Y_turbo.min())
            model = GP_model(DB._X, scaled_y)
    
            M1_time = time()
            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
                model.custom_fit(Global_Var.large_fit)
                t_model = time ()
                time_per_cycle[iter, 0] = t_model - t_start
                
                # Create a batch
                X_next = Turbo.generate_batch(mod = model, batch_size = batch_size, acqf = "ei")
                t_ap = time()
                time_per_cycle[iter, 1] = t_ap - t_model
                
            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(X_next)):
                send_to = c%n_proc
                n_cand[send_to] += 1
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(X_next)):
                send_cand = X_next[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)

            ## Evaluate
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
                else:
                    Y_next[c] = torch.tensor(y_new[k])
                    DB.add(X_next[c].unsqueeze(0), torch.tensor(y_new[k]).unsqueeze(0))
                    k =+ 1
            M2_time = time()
            
            # Update state
            Turbo.update_state(-Y_next)
            print("Alg. TuRBO, cycle ", iter, " took --- %s seconds ---" % (M2_time - M1_time))
            print('Best known target is: ', torch.min(DB._y).numpy())

    
            target[iter+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[iter, 2] = t_ap - t_start # t_ap + t_model
            time_per_cycle[iter, 3] = t_end - t_ap # t_sim
            time_per_cycle[iter, 4] = t_end - t_start # t_tot
            if(sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[iter, 2] + sim_cost

#            t_current = time() - t_0
            iter = iter + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)
    
        print('Final MC_q-EGO value :', end='')
        DB.print_min()
        del DB

    else:
        DB_worker = DataBase(f, n_init) # in order to access function evaluation
        DB_worker.par_create(comm = comm)

        for iter in range(n_cycle+1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)
            if (n_cand.sum() == 0):
                break
            else :
                ## Receive from master 
                cand = []
                for c in range(n_cand[my_rank]):
                    cand.append(comm.recv(source = 0, tag = my_rank + c * n_proc))
    
                ## Evaluate
                y_new = []
                for c in range(n_cand[my_rank]):
                    y_new.append(DB_worker.eval_f(cand[c]))
    
                ## Send it back
                for c in range(n_cand[my_rank]):
                    comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
                
        del DB_worker

if __name__ == "__main__":
    main()
