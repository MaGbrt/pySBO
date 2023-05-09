#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:27:08 2022

@author: maxime
"""
import torch
import numpy as np
import operator

from gpytorch.settings import cholesky_jitter
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import Global_Var
from Global_Var import *
from time import time
import random

class Subdomains(): 
    def __init__(self):
        self._list = [];
        self._size = -1;
        
    def select(self, k):
        n = self.get_size()
        while (n > k):
            idx = random.randint(0,n-1)
            del self._list[np.array(idx)]
            n -= 1
    def select_(self, n):
        k = 0
        cpt = 0
        while(self.get_size() > n):
#            print('Index is ' + str(k) + ' and list size is ' + str(self.get_size()))
            if (torch.rand(1)>0.5):
                 del self._list[k]
#                 print('Remove index ' + str(k) + ' size is ' + str(self.get_size()))
                 if (k == self.get_size()):
                     k = 0
            else:
                if (k < self.get_size()-1):
                    k += 1
                else:
                    k = 0
            cpt += 1
            
            
    def local_APs(self, n_cand, model):
        X_next = []
        AF_values = []
        crit = ExpectedImprovement(model._model, torch.min(model._train_Y), maximize = False)
        for elmt in self._list:
#            print(elmt._domain)
            with cholesky_jitter(parameters.chol_jitter):
                candidate, alpha = optimize_acqf(crit, bounds=elmt._domain, q=1, num_restarts=Global_Var.af_nrestarts, raw_samples=Global_Var.af_nsamples, options=Global_Var.af_options)

            elmt._crit = alpha.numpy()
            AF_values.append(alpha.numpy())
            X_next.append(candidate[0].numpy())
        
        try :
            sort_X = [el for _, el in sorted(zip(AF_values, X_next), reverse = True)]
        except :
            print('Failed to sort list')
            sort_X = X_next
        #print(sort_X)
        return sort_X
    
    def sort_list(self, decreasing = True):
        new_list = sorted(self._list, key = operator.attrgetter('_crit'), reverse = decreasing)
        return new_list

    def sort_list_indexes(self, decreasing = True):
        new_list = sorted(self._list, key = operator.attrgetter('_index'), reverse = decreasing)
        return new_list
    
    def print_indexes(self):
        for elmt in self._list:
            print(elmt._index, end = ' ')
        print(' ')
        
    def get_size(self):
        size = 0
        for elmt in self._list:
            size += 1
        self._size = size
#        print('My list has ', self._size, ' subdomains')
        return size