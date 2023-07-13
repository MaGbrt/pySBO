#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:57:40 2022

@author: maxime
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:46:58 2022

@author: maxime
"""

import numpy as np
from time import time
import torch


from botorch.optim import optimize_acqf
dtype = torch.double


def par_random_run(DB, n_cycle, t_max, batch_size, id_run, comm = None):
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
            
            t_model = time ()
            time_per_cycle[iter, 0] = t_model - t_start

            # Acquisition process
            candidates = torch.rand(batch_size, DB._dim)
            
            t_ap = time()
            time_per_cycle[iter, 1] = t_ap - t_model

            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1
            
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(candidates)):
                send_cand = candidates[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
            
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

            t_end = time()
            time_per_cycle[iter, 2] = t_ap - t_start
            time_per_cycle[iter, 3] = t_end - t_ap
            time_per_cycle[iter, 4] = t_end - t_start
            target[0, iter+1] = torch.min(DB._y).numpy()
            print("Alg. Random, cycle ", iter, " took --- %s seconds ---" % (t_end - t_start))
            print('Best known target is: ', torch.min(DB._y).numpy())
            
            iter = iter + 1
            t_current = time() - t_0
        else :
            print('Budget is out, time is %.6f' %t_current)
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)
        
        
        print('Final random value :', torch.min(DB._y).numpy())
        DB.print_min()
        return target, time_per_cycle


    else:
        for iter in range(n_cycle+1):
            ## Get number of candidates to compute
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)

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
    