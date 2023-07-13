#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:36:36 2023

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
    

def par_Turbom_run(DB, n_cycle, t_max, batch_size, m, acq_f = "ei", id_run = None, comm = None):
    my_rank = comm.Get_rank()
    n_proc = comm.Get_size()
    n_steps = 1 # number of local turbo steps before syncing
    if my_rank == 0 :
        print('Best initial known target is: ', torch.min(DB._y).numpy())
        target = np.zeros((1, n_cycle+1))
        target[0, 0] = DB._y.min()
        time_per_cycle = np.zeros((n_cycle, 5))

        # List of points - gives the value of a TR
        rank = np.zeros(m)
        gain = np.zeros(m, dtype = bool)
        # List of turbo objects
        turbobj_list = []
        # List of DB
        DB_list = []

        n_pts = int (np.ceil(DB._size*2/3))
        for k in range(m):
            turbobj_list.append(TuRBO(DB._dim, 1))
            DB_list.append(DB.create_subset(n_pts)) 


        # Local TuRBO algs
        Turbobj = TuRBO(DB._dim, 1)

        itr = 0
        t_0 = time()
        t_current = 0.
        t_start = time()
        while (itr < n_cycle and t_current < t_max):
            n_cand = np.ones(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)

            M1_time = time()
            # Master decides which TuRBO object will be run
            for k in range(m):
                if(turbobj_list[k].restart_triggered == True):
                    DB_list[k] = DB.create_subset(n_pts + itr)
                    gain[k] = 0
                    print('Restart TR n°: ', k)
            # Selection of q TR to trigger # TR with the highest success rate (the rank)
            selected_TR = np.argpartition(rank, -batch_size)[-batch_size:]
            
            # Send TuRBO object to workers ;  Send DB
            send_to = 1
            for selec in selected_TR[1:]:
                comm.send(DB_list[selec]._X.numpy(), dest = send_to, tag = send_to + 1000)
                comm.send(DB_list[selec]._y.numpy(), dest = send_to, tag = send_to + 2000)
                comm.send(turbobj_list[selec].state_to_vec(), dest = send_to, tag = send_to + 3000)
                send_to += 1
    
            # All: initialise turbostate
            Turbobj.vec_to_state(turbobj_list[selected_TR[0]].state_to_vec())
            
            # All : trigger a step of TuRBO in the TR
            loc_gain = TuRBO_step(DB_list[selected_TR[0]], Turbobj, 1, n_steps, acq_f = "ei") # returns 1 if gain
            
            # Synchronizing with master
            gain[selected_TR[0]] = loc_gain
            DB.add(DB_list[selected_TR[0]]._X[-1], DB_list[selected_TR[0]]._y[-1].unsqueeze(0)) # supplement global DB
            turbobj_list[selected_TR[0]].vec_to_state(Turbobj.state_to_vec())
            
            # Gather new points
            # Receive info from turbobj
            recv_from = 1
            for selec in selected_TR[1:]:        
                x_temp = comm.recv(source = recv_from, tag = recv_from + 5000)
                y_temp = comm.recv(source = recv_from, tag = recv_from + 6000)
                DB_list[selec].add(torch.tensor(x_temp), torch.tensor(y_temp).unsqueeze(0)) # supplement local DB
                DB.add(torch.tensor(x_temp), torch.tensor(y_temp).unsqueeze(0)) # supplement global DB
                turbobj_list[selec].vec_to_state(comm.recv(source = recv_from, tag = recv_from + 7000))
                gain[selec] = comm.recv(source = recv_from, tag = recv_from + 8000)
                recv_from += 1
    
            for k in range(m):
                if gain[k] == 0:
                    rank[k] -= 1
            print('Rank of TR: ', rank)
            M2_time = time()
            print("Alg. TuRBO, cycle ", itr, " took --- %s seconds ---" % (M2_time - M1_time))
            print('Best known target is: ', torch.min(DB._y).numpy())
            target[0, itr+1] = torch.min(DB._y).numpy()
            t_end = time()
            time_per_cycle[itr, 1] = 0 #t_ap - t_model
            time_per_cycle[itr, 2] = 0 #t_ap - t_start # t_ap + t_model
            time_per_cycle[itr, 3] = 0 #t_end - t_ap # t_sim
            time_per_cycle[itr, 4] = M2_time - M1_time # t_cyc
            if(Global_Var.sim_cost == -1): # real cost
                t_current = time() - t_0
            else:
                t_current += time_per_cycle[itr, 2] + Global_Var.sim_cost
            print('t_current: ', t_current)
            print('real time is ', time() - t_0)

    #            t_current = time() - t_0
            itr = itr + 1
        else :
            print('Budget is out, time is %.6f' %t_current)
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)

        DB.print_min()
        return target, time_per_cycle

    else : # workers
        # Local TuRBO algs
        Turbobj = TuRBO(DB._dim, 1)

        for itr in range(n_cycle + 1):
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)
            if (n_cand.sum() == 0):
                break
            else :
               # Receive info about the triggered TR
                X_ = comm.recv(source = 0, tag = my_rank + 1000)
                y_ = comm.recv(source = 0, tag = my_rank + 2000)
                DB.set_Xy(X_, y_)
                # All: initialise turbostate
                Turbobj.vec_to_state(comm.recv(source = 0, tag = my_rank + 3000))
            
                # All : trigger a step of TuRBO in the TR
                loc_gain = TuRBO_step(DB, Turbobj, 1, n_steps, acq_f = "ei") # returns 1 if gain
                comm.send(DB._X[-1].numpy(), dest = 0, tag = my_rank + 5000) # send back new cand
                comm.send(DB._y[-1].numpy(), dest = 0, tag = my_rank + 6000) # send back new cand
                comm.send(Turbobj.state_to_vec(), dest = 0, tag = my_rank + 7000) # send back updated TR
                comm.send(loc_gain, dest = 0, tag = my_rank + 8000) # send back gain

        return None


                
def TuRBO_step(DB, turbobj, batch_size, n_steps, acq_f = "ei"):
    # Given a data base, the TuRBO step performs an acquisition process and increment de DB.
    temp_best = DB._y.min()
    gain = False
    for i in range(n_steps):
        Y_turbo = -DB._y.clone().detach()
        scaled_y = (Y_turbo - Y_turbo.min()) / (Y_turbo.max()-Y_turbo.min())
        model = GP_model(DB._X, scaled_y)
        
        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
            model.custom_fit(Global_Var.large_fit)
            X_next = turbobj.generate_batch(mod = model, batch_size = 1, acqf = acq_f)
            y_new = DB.eval_f(X_next.numpy())
            DB.add(X_next, torch.tensor(y_new))
                
            turbobj.update_state(-y_new)
        if y_new[0] < temp_best.numpy() :
            gain = True
    return gain

