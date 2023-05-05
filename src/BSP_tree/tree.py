#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:27:17 2022

@author: maxime
"""
import torch
import numpy as np
from BSP_tree.split_functions import default_split
from DataSets.DataSet import DataBase
from BSP_tree.subdomains import Subdomains
from time import time

class Tree(DataBase): 
    def __init__(self, DB=None, domain=None):
        self._parent = None;
        self._left = None;
        self._right = None;
        
        self._index = 0;
        self._level = 0;
        self._domain = domain;
        self._crit = -1.; # Crit to sort the leaves
        self._points = DB;     
    
    def __del__(self):
        del self._index
        del self._level
        del self._domain
        del self._crit
        del self._points     
        del self._left
        del self._right
        
    #%% Building functions
    def build(self, depth, split_function = default_split):
        if (depth > 0):
            self.spread_node(split_function)
            self._left.build(depth-1, split_function)
            self._right.build(depth-1, split_function)

    def spread_node(self, split_function = default_split): # Build 1 more stage starting from current node
        self.spread_left(split_function)
        self.spread_right(split_function)
        
    def spread_left(self, split_function):
        self._left = Tree()
        self._left._parent = self
        self._left._index = self.cp_index_left()
        self._left._level = self._level + 1
        self._left._domain = split_function(self._domain, 0)
        self._left._crit = self._crit
        
    def spread_right(self, split_function):
        self._right = Tree()
        self._right._parent = self
        self._right._index = self.cp_index_right()
        self._right._level = self._level + 1
        self._right._domain = split_function(self._domain, 1)
        self._right._crit = self._crit
        
    def cut_leaves(self):
        assert self._right != None and self._left != None, 'You are not cutting leaves, tree is compromised'
        del self._left
        del self._right
        self._left = None
        self._right = None
        
    #%% Evolution of the tree
    def propagate_crit(self): # Bad performances
        a = 0.0001
#        print('I am node ', self._index, ' and I get ', self._left._crit, ' from left child and ', self._right._crit, ' from right child')                                                                
        if (self._left.is_leaf()):
            if (self._right.is_leaf()):
                b = max(self._left._crit, self._right._crit)
                self._crit = b + abs(b)*a
            else:
                self._right.propagate_crit()
                b = max(self._left._crit, self._right._crit)
                self._crit = b + abs(b)*a
        else:
            self._left.propagate_crit()
            b = max(self._left._crit, self._right._crit)
            self._crit = b + abs(b)*a

        if (self._right.is_leaf()):
                b = max(self._left._crit, self._right._crit)
                self._crit = b + abs(b)*a
        else:
            self._right.propagate_crit()
            b = max(self._left._crit, self._right._crit)
            self._crit = b + abs(b)*a

    def propagate_crit_2(self, subdomains):
        a=0.0001
        if (subdomains.get_size() != 1 or subdomains._list[0]._index != 0):
            parents = Subdomains()
            for sbdmn in subdomains._list:
                ok = True;
                if sbdmn._parent:
                    idx = sbdmn._parent._index;
                    for p in parents._list:
                        if (p._index == idx):
                            ok = False
                            break
                    if ok :
                        parents._list.append(sbdmn._parent)
                
            parents._list = parents.sort_list_indexes(decreasing = True) # starting from deeper indexes
 #           parents.print_indexes()
            for p in parents._list:
                if p._parent == None :
                    continue
                else :
                    b = max(p._left._crit, p._right._crit)
                    p._crit = b + abs(b)*a
            self.propagate_crit_2(parents)
        else :
            for sbdmn in subdomains._list:
                b = max(sbdmn._left._crit, sbdmn._right._crit)
                sbdmn._crit = b + abs(b)*a
#            print('Propagation done')
    #%% Update function
    def update(self, subdomains):
        assert self._parent == None, 'propagate_crit must be called from root'
        assert self.is_leaf() == False, 'If root is a leaf, then run EGO'

        t_prop = time()
        self.propagate_crit_2(subdomains)
#        print('t_propagate2 = ', time() - t_prop)

        subdomains._list = subdomains.sort_list(decreasing = True)
#        subdomains.print_indexes()
#        print('Best leaf is ', subdomains._list[0]._index)
        
        parents = Subdomains()
        max_lvl = 0
        for sbdmn in subdomains._list:
            parents._list.append(sbdmn._parent)
            max_lvl = np.max([max_lvl, sbdmn._level])
        print('max lvl = ', max_lvl)

        
        parents._list = parents.sort_list(decreasing = False)
#        print('Worst parent is ', parents._list[0]._index)

        if(subdomains._list[0]._left == None and subdomains._list[0]._right == None): # if best is leaf
            if (parents._list[0]._left != None and parents._list[0]._right != None): # if worst is not leaf
#                print('Spread node ', subdomains._list[0]._index)
#                print('Cutting leaves ', parents._list[0]._left._index, ' and ', parents._list[0]._right._index)
                if (subdomains._list[0]._index != parents._list[0]._left._index and subdomains._list[0]._index != parents._list[0]._right._index):
                    subdomains._list[0].spread_node()

                    # replace best by children
                    subdomains._list.append(subdomains._list[0]._left)
                    subdomains._list.append(subdomains._list[0]._right)
                    subdomains._list.remove(subdomains._list[0])
                    parents._list[0].cut_leaves()
                else:
                    print('Cannot proceed')
        self.check_volume()
        #self.print_tree()
        
    def update_split_only(self, subdomains):
        assert self._parent == None, 'propagate_crit must be called from root'
        assert self.is_leaf() == False, 'If root is a leaf, then run EGO'

        t_prop = time()
        self.propagate_crit_2(subdomains)
#        print('t_propagate2 = ', time() - t_prop)

        subdomains._list = subdomains.sort_list(decreasing = True)
#        subdomains.print_indexes()
#        print('Best leaf is ', subdomains._list[0]._index)
        
        if(subdomains._list[0]._left == None and subdomains._list[0]._right == None): # if best is leaf
            subdomains._list[0].spread_node()
        else:
            print('Cannot proceed')
        self.check_volume()
        #self.print_tree()
        
    def update_twice(self, subdomains):
        print('Function not up to date, might be too much time consuming for large trees')
        assert self._parent == None, 'propagate_crit must be called from root'
        assert self.is_leaf() == False, 'If root is a leaf, then run EGO'
        

        self.propagate_crit()
        subdomains._list = subdomains.sort_list(decreasing = True)
#        print('Best leaf is ', subdomains._list[0]._index)
        
        parents = Subdomains()
        for sbdmn in subdomains._list:
            parents._list.append(sbdmn._parent)

        parents._list = parents.sort_list(decreasing = False)
        #print('Worst parent is ', parents._list[0]._index)
        if(subdomains._list[0]._left == None and subdomains._list[0]._right == None): # if best is leaf
            if (parents._list[0]._left != None and parents._list[0]._right != None): # if worst is not leaf
                # print('Spread node ', subdomains._list[0]._index)
                # print('Cutting leaves ', parents._list[0]._left._index, ' and ', parents._list[0]._right._index)
                if (subdomains._list[0]._index != parents._list[0]._left._index and subdomains._list[0]._index != parents._list[0]._right._index):
                    subdomains._list[0].spread_node()
                    # replace best by children
                    subdomains._list.append(subdomains._list[0]._left)
                    subdomains._list.append(subdomains._list[0]._right)
                    subdomains._list.remove(subdomains._list[0])
                    parents._list[0].cut_leaves()
                else:
                    print('Cannot proceed')
        self.check_volume()
        self.update(self.get_leaves())
        
        
        
    #%% Acquisition Process functions
    def get_leaves(self):
        assert self._parent == None, 'get_leaves must be called from root'

        leaves = Subdomains()
        self.roam_tree(leaves)

        return leaves

    def roam_tree(self, leaves):
        if (self.is_leaf()):
            leaves._list.append(self)
        else:
            self._left.roam_tree(leaves)
            self._right.roam_tree(leaves)
            
    #%% Utils functions
    def cp_index_left(self):
        return 2 * self._index + 1;

    def cp_index_right(self):
        return 2 * self._index + 2;
    
    def cp_index_parent(self):
        if(self._index != 0):
            return (self._index - 1) / 2;
        else:
            return 0;
        
    def get_brother(self):
        if (self._parent != None):
            if (self._index == self._parent._left._index):
                return self._parent._right
            else:
                return self._parent._left
        else:
            return None
        
    def is_leaf (self):
        if (self._left == None and self._right == None):
            return True;
        else:
            return False;

    def check_volume (self):
        leaves = self.get_leaves()
        V = 0.
        for l in leaves._list:
            V += torch.prod(l._domain[1]-l._domain[0], dtype=torch.float64)
#        print('Volume is: ', V)
        assert V==1, ('Check the split function, total volume is', V.numpy(), ' not 1')
        
    #%% Print functions 
    def print_node(self):
        print('I am node: ', self._index)
        print('My subdomain is: ', self._domain)
        if (self._index != 0):
            print('Parent index is: ', self._parent._index)
            print('Left child index is: ', self._left._index)
            print('Right child index is: ', self._right._index)      
            
    def print_tree(self):
        for i in range(self._level):
            print("| _ _", end='')
        print(self._index, "{:.4f}".format(self._crit))

#        if (self._left == None and self._right == None):
#            print(self._index, self._domain[0], self._domain[1])
#            print(self._index, self._crit)
        if (self._left != None and self._right != None):
            self._left.print_tree();
            self._right.print_tree();
   