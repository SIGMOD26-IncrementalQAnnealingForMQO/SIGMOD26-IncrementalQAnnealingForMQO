import numpy as np
import itertools


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
        
    #dwave_qubo = dimod.as_bqm(qubo_matrix.tolist())
    
    #qubo_matrix = np.triu(qubo_matrix)
    
    bqm = dimod.BinaryQuadraticModel(qubo_matrix, 'BINARY')
    return bqm

    
def generate_Fujitsu_QUBO_with_matrix(queries, costs, qubo_matrix, wl, wm):
    from dadk.BinPol import BinPol
    import time

    start_time = time.time()

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