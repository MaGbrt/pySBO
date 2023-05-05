#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:29:26 2022

@author: maxime
"""
import random
import torch


def default_split(domain, upper):
    d_split = int(((domain[1]-domain[0])).argmax().numpy())
#    print('Split dimension is ', d_split)
    a = 0 # random.random()/50
    subdomain = torch.clone(domain)
    if (upper == 0):
        mid = torch.clone(domain[1])
        mid[d_split] = (domain[0][d_split] + domain[1][d_split]) * (0.5 + a)
        subdomain[1] = mid
#        print('Lower bound', subdomain)
    elif (upper == 1):
        mid = torch.clone(domain[0])
        mid[d_split] = (domain[0][d_split] + domain[1][d_split]) * (0.5 + a)
        subdomain[0] = mid
#        print('upper bound', subdomain)
    return subdomain