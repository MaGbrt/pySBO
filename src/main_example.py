#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example for running TuRBO, gBSP-EGO and lBSP-EGO

mpiexec -n 2 python3 main_example.py

"""
from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_proc = comm.Get_size()
print('From ', my_rank, ' : Running main with ', n_proc, 'proc')
import numpy as np
import sys
import Global_Var
from Global_Var import *

from Problems.Rosenbrock import Rosenbrock
from Problems.Alpine02 import Alpine02
from Problems.Rastrigin import Rastrigin
from Problems.Ackley import Ackley
from Problems.Schwefel import Schwefel
from Problems.CEC2014 import CEC2014
from Problems.Hybrid_AW import Hybrid_AW
from Problems.Composition_AS import Composition_AS
from Problems.enoppy_plant_problem import Plant
from Problems.enoppy_bearing_problem import Bearing

from Full_loops.parallel_EGO_cycle import par_KB_qEGO_run, par_MIC_qEGO_run, par_qEGO_run, par_Lanczos_qEGO_run, par_TS_qEGO_run
from Full_loops.parallel_SAASBO import par_SAASBO_run
from Full_loops.parallel_g1_BSP_EGO_cycle import par_g1_BSP_EGO_run, par_g1_BSP_qEGO_run, par_g1_BSP2_EGO_run
from Full_loops.parallel_l1_BSP_EGO_cycle import par_l1_BSP_EGO_run, par_lg1_BSP_EGO_run, par_lg2_BSP_EGO_run, par_l2_BSP_EGO_run
from Full_loops.parallel_SKA_EGO_cycle import par_MCbased_SKIP_qEGO_run, par_MCbased_Sparse_qEGO_run
from Full_loops.parallel_SAGA_SaaF import par_SAGA_SaaF_run
from Full_loops.parallel_SAPSO_SaaF import par_SAPSO_SaaF_run
from Full_loops.parallel_TuRBO1_cycle import par_Turbo1_run, par_fast_Turbo1_run
from Full_loops.parallel_TuRBOm_cycle import par_Turbom_run
from Full_loops.parallel_eShotgun import par_eShotgun_run
from Full_loops.parallel_ABAFMo import par_ABAFMo_run
from Full_loops.parallel_MACE import par_MACE_run
from Full_loops.parallel_Hybrid_TuRBO_SAGA import par_Hybrid_TuRBO_SAGA_run
from Full_loops.parallel_random_sampling import par_random_run
from Full_loops.parallel_GA import par_GA_run
from Full_loops.parallel_PSO import par_PSO_run

from DataSets.DataSet import DataBase
from random import random
 
# Budget parameters
DoE_num = 0; 
dim = 6;
batch_size = n_proc;
budget = 10;
t_max = 1000; # seconds
n_init = 60; #(int) (min((0.2*budget)*batch_size, 128));
n_cycle = 10; #(int) (0.8*budget);

size_Lanczos = Global_Var.size_Lanczos


if my_rank == 0:
    print('The budget for this run is:', budget, ' cycles.')
n_leaves = 4*n_proc
tree_depth = int(np.log(n_leaves)/np.log(2))
#n_init_leaves = pow(2, tree_depth)
#max_leaves = 4 * n_init_leaves
n_learn = min(n_init, 128)
n_TR = 2 * batch_size

threshold = Global_Var.threshold

# Define the problem
folder = 'Results/'
ext = '.txt'
f = Ackley(dim)
id_name = '_' + 'Ackley' + '_D' + str(dim) + '_batch' + str(batch_size) + '_budget' + str(budget) + '_t_cost' + str(Global_Var.sim_cost)

if (my_rank == 0):
    target = np.zeros((1, n_cycle+1))

    # Input data scaled in [0, 1]^d
    DB = DataBase(f, n_init)
    r = random()*1000
    par_create = np.ones(1, dtype = 'i')
    DB.par_create(comm = comm, seed = r)
    full_name = 'Initial_DoE' + id_name + '_run_test' + ext
    DB.save_txt('DoE/' + full_name)
    
    comm.Barrier()
    
    print('\n Synchronize before running Turbo')
    DB_Turbo_ei = DB.copy()
    target[0, :], time_turbo_ei = par_Turbo1_run(DB_Turbo_ei, n_cycle, t_max, batch_size, "ei", DoE_num, comm)
    full_name = 'TuRBO' + id_name + '_t_max' + str(t_max) + '_run_test' + ext
    DB_Turbo_ei.save_txt(folder + full_name)
    np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_turbo_ei, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
    del DB_Turbo_ei

    
    # comm.Barrier()
    # print('\n Synchronize before running BSP-EGO with global model')
    # DB_g1_BSP_EGO = DB.copy()
    # target[0, :], time_g1_BSP_EGO = par_g1_BSP_EGO_run(DB_g1_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, DoE_num, comm)
    # full_name = 'Global_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run_test' + ext
    # DB_g1_BSP_EGO.save_txt(folder + full_name)
    # np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_g1_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
    # del DB_g1_BSP_EGO
            
    # comm.Barrier()
    # print('\n Synchronize before running l2BSP-EGO with local models')
    # DB_l2_BSP_EGO = DB.copy()
    # target[0, :], time_l2_BSP_EGO = par_l2_BSP_EGO_run(DB_l2_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, n_learn, DoE_num, comm)
    # full_name = 'L2_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run_test' + ext
    # DB_l2_BSP_EGO.save_txt(folder + full_name)
    # np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_l2_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
    # del DB_l2_BSP_EGO

        
else:
    DB_worker = DataBase(f, n_init) # in order to access function evaluation
    DB_worker.par_create(comm = comm)
    
    comm.Barrier()
    print('\n Synchronize before running turbo - workers')
    DB_worker = DataBase(f, n_init) # in order to access function evaluation
    par_Turbo1_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
    del DB_worker

    # comm.Barrier()
    # print('\n Synchronize before running BSP-EGO with global model - workers')
    # DB_worker = DataBase(f, n_init) # in order to access function evaluation
    # par_g1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
    # del DB_worker

    # comm.Barrier()
    # print('\n Synchronize before running l2BSP-EGO with local models and one global model - workers')
    # DB_worker = DataBase(f, n_init) # in order to access function evaluation
    # par_l2_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
    # del DB_worker
