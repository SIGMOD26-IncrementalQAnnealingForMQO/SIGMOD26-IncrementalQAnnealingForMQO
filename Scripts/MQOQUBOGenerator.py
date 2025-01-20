#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import unittest

import json
import os
import pickle
import pathlib 
import csv


# In[2]:


def calculate_wl(costs, epsilon):
    return max(costs)+epsilon

def calculate_wm(num_plans, savings, wl):
    if not savings:
        return wl
    max_savings_for_plan = np.zeros(num_plans)
    for ((i, j), s) in savings.items():
        max_savings_for_plan[i] = max_savings_for_plan[i] + s
        max_savings_for_plan[j] = max_savings_for_plan[j] + s
    return wl + max(max_savings_for_plan)


# In[ ]:
    
def generate_DWave_QUBO_with_matrix(queries, costs, qubo_matrix, wl, wm):
    import dimod
    num_plans = len(costs)
    
    epsilon = 0.25
    
    for (q, plans) in queries.items():
        for (p1, p2) in itertools.combinations(plans, 2):
            qubo_matrix[p1][p2] = wm
            qubo_matrix[p2][p1] = wm
    for i in range(num_plans):
        qubo_matrix[i][i] = costs[i]-wl
        
    bqm = dimod.BinaryQuadraticModel(qubo_matrix, 'BINARY')
    return bqm
    
def generate_NEC_QUBO_with_matrix(queries, costs, qubo_matrix, wl, wm):
    import VectorAnnealing
    
    num_plans = len(costs)
    
    epsilon = 0.25
    
    qubo = {}
    query_counter = 0
    one_hot_list = []
        
    for (q, plans) in queries.items():
        one_hot = [str(plan) for plan in plans]
        one_hot_list.append(one_hot)
        for (p1, p2) in itertools.combinations(plans, 2):
            qubo_matrix[p1][p2] = wm
            qubo_matrix[p2][p1] = wm
    for i in range(num_plans):
        qubo_matrix[i][i] = costs[i]-wl
        
    for i in range(num_plans):
        for j in range(num_plans):
            if j < i:
                continue
            if qubo_matrix[i][j] != 0:
                qubo[(str(i), str(j))] = float(qubo_matrix[i][j])
    
    problem_QUBO = VectorAnnealing.model(qubo, 0, onehot=one_hot_list)
    return problem_QUBO
    

    
def generate_Fujitsu_QUBO_with_matrix(queries, costs, qubo_matrix, wl, wm):

    from dadk.BinPol import BinPol

    num_plans = len(costs)
    
    epsilon = 0.25
    
    for (q, plans) in queries.items():
        for (p1, p2) in itertools.combinations(plans, 2):
            qubo_matrix[p1][p2] = wm
            qubo_matrix[p2][p1] = wm
    for i in range(num_plans):
        qubo_matrix[i][i] = costs[i]-wl
    
    fujitsu_qubo = BinPol(qubo_matrix_array=qubo_matrix)
    return fujitsu_qubo