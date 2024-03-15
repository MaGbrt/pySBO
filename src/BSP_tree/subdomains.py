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
        
    def select_leaves(self, k, crit):
        decreasing = False
        
        n = self._size
        # print('List is of size ', n)
        # print('We keep only ', k, ' leaves')
        # Sort leaves according to the best candidate in each sub-region
        if crit == 'value':
            sort_sbdmns = sorted(self._list, key = operator.attrgetter('_best'), reverse = decreasing)
        elif crit == 'af':
            sort_sbdmns = sorted(self._list, key = operator.attrgetter('_crit'), reverse = decreasing)
        self._list = sort_sbdmns
        print('Length of sorted list ', len(sort_sbdmns))
        print('Selected subdomains:')
        indexes = np.zeros(n, dtype=object)
        for d in range(n):
            indexes[d] = sort_sbdmns[d]._index
            print(sort_sbdmns[d]._index, self._list[d]._index)
        # print('Removing:')
        new_list = []
        cpt = 0
        for cpt in range(k):
            prob = len(new_list)/k #first always taken, proba decreases
            a = torch.rand(1)
            itr = 0
            while a < prob:
                itr += 1
                a = torch.rand(1)
                if itr >= len(self._list):
                    itr = 0
                continue
            else:
                new_list.append(self._list[itr])
                del self._list[itr]
        self._list = new_list
        # while self.get_size() > k:
        #     if self._list[k]._index != min(indexes) :
        #         # print('idx ', self._list[k]._index)
        #         del self._list[k] # Remove always the last one (list is modified each time, then also the indexes)
        #     else:
        #         # print('idx ', self._list[k-1]._index)
        #         del self._list[k-1]
        print('Remaining: ')
        for leaf in self._list:
            print('Leaf: ', leaf._index, ' crit: ', leaf._crit, ' best ', leaf._best)
        self._size = self.get_size()


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
    
    def sort_list(self, decreasing = True, according_to = '_crit'):
        new_list = sorted(self._list, key = operator.attrgetter(according_to), reverse = decreasing)
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