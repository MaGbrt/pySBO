#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:47:52 2023

@author: maxime
"""

from numpy import ma
import numpy as np
import matplotlib.pyplot as plt
from time import time

from Problems.Single_Objective import Single_Objective

from Problems.enoppy_main.enoppy.paper_based import rwco_2020

#------------------------------------------------------#
#------------- class enoppy_plant_problem -------------#
#------------------------------------------------------#


class Plant(Single_Objective):
    """Class for the single-objective Multi-Product Batch Plant problem.
    
    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#    
    def __init__(self, n_dvar):
        assert n_dvar==10, 'The Multi-Product Batch Plant problem is only define for 10 decision variables'
        self._enoppy_pb = rwco_2020.MultiProductBatchPlantProblem()
        Single_Objective.__init__(self, n_dvar)

    #-------------__del__-------------#
    def __del__(self):
        Single_Objective.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "Multi-Product Batch Plant problem "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objective"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_real_evaluation-------------#
    def perform_real_evaluation(self, candidates):
        """Objective function.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :return: objective values
        :rtype: np.ndarray
        """
        start = time()
        assert self.is_feasible(candidates)

        
        if candidates.ndim==1:
            x = self._enoppy_pb.amend_position(candidates)
            y = self._enoppy_pb.evaluate(x)
            p = y

        else:
            p = np.zeros(len(candidates))
            #print('I have ', len(candidates), ' candidates to evaluate')
            k = 0
            for cand in candidates:
                x = self._enoppy_pb.amend_position(cand)
                y = self._enoppy_pb.evaluate(x)
                p[k] = y[0]
                k += 1
                
        end = time()
#        print('Evaluation done, returns: ', p, ' in ', end-start, ' s.')
        return p


    #-------------get_bounds-------------#
    def get_bounds(self):
        """Returns search space bounds.

        :returns: search space bounds
        :rtype: np.ndarray
        """
        b = self._enoppy_pb.bounds.transpose()
        
        return b

    #-------------is_feasible-------------#
    def is_feasible(self, candidates):
        """Check feasibility of candidates.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :returns: boolean indicating whether candidates are feasible
        :rtype: bool
        """

        res=False
        if Single_Objective.is_feasible(self, candidates)==True:
            lower_bounds=self.get_bounds()[0,:]
            upper_bounds=self.get_bounds()[1,:]
            res=(lower_bounds<=candidates).all() and (candidates<=upper_bounds).all()
        return res
    #-------------plot-------------#
    def plot(self):
        print("[Plant.py] Impossible to plot Plant")
