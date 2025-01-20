#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import os
import pickle
import pathlib
import itertools

import Scripts.MQOQUBOGenerator as MQOQUBOGenerator
import Scripts.MQOProblemGenerator as MQOProblemGenerator
import Scripts.BenchmarkProblemGenerator as BenchmarkProblemGenerator

import Scripts.PartitioningQUBOGenerator as PartitioningQUBOGenerator
import Scripts.DataExport as DataExport
import Scripts.DataUtil as DataUtil
import Scripts.DWaveUtil as DWaveUtil
import time
from math import inf
import neal
import dimod

# In[2]:


def get_name_for_solver_settings(solver_settings):
    if solver_settings["solver"] == 'DA':
        return 'DA_tl_' + str(solver_settings["time_limit_sec"]) + '_npart_' + str(solver_settings["num_partitioning_phases"])
    elif solver_settings["solver"] == 'SA':
        return "SA_it_" + str(solver_settings["number_iterations"]) + '_npart_' + str(solver_settings["num_partitioning_phases"])
    elif solver_settings["solver"] == 'DWave_Hybrid':
        return "DWave_Hybrid_tl" + str(solver_settings["time_limit"]) + '_npart_' + str(solver_settings["num_partitioning_phases"])
    elif solver_settings["solver"] == 'SQA':
        return "SQA_it_" + str(solver_settings["n_steps"]) + '_npart_' + str(solver_settings["num_partitioning_phases"])
    elif solver_settings["solver"] == 'VA':
        return "VA_mode_" + solver_settings["vector_mode"] + "_it_" + str(solver_settings["num_sweeps"]) + '_npart_' + str(solver_settings["num_partitioning_phases"])
    else:
        return ""
    
def export_result(plan_configuration, costs, solution_time, solver_settings, result_path):
    solution_time_in_ms = solution_time*1000
    result = {"time_in_ms": solution_time_in_ms, "costs": costs, "plan_selection": plan_configuration}
    result_name = get_name_for_solver_settings(solver_settings) + '.json'
    DataUtil.save_data(result, result_path, result_name, override=False)
    
def check_if_result_exists(solver_settings, result_path):
    result_name = get_name_for_solver_settings(solver_settings) + '.json'
    if os.path.exists(result_path + '/' + result_name):
        return True
    return False


# In[3]:


def evaluate_mqo_solution(active_plans, plan_costs, savings_matrix, num_og_plans):

    active_savings = 0
    
    active_plans_solution = []
    
    for (p1, p2) in itertools.combinations(active_plans, 2):
        active_savings += savings_matrix[p1][p2]
        active_plans_solution.append(p1)
        active_plans_solution.append(p2)
            
    active_plans_solution = sorted(set(active_plans_solution))
    
    costs = sum(plan_costs[x] for x in active_plans_solution)

    return costs - active_savings


def generate_partitioning_encoding_with_qubo_matrix(qubo_matrix, solver_settings):
    if solver_settings["solver"] == "DA":
        from dadk.BinPol import BinPol
        partitioning_encoding = BinPol(qubo_matrix_array=qubo_matrix)
    elif solver_settings["solver"] == "SA":
        partitioning_encoding = dimod.BinaryQuadraticModel(qubo_matrix, 'BINARY')
    elif solver_settings["solver"] == "DWave_Hybrid":
        partitioning_encoding = dimod.BinaryQuadraticModel(qubo_matrix, 'BINARY')
    elif solver_settings["solver"] == "SQA":
        from qat.opt import QUBO
        partitioning_encoding = QUBO(Q=qubo_matrix)
    elif solver_settings["solver"] == "VA":
        import VectorAnnealing
        qubo = {}
        num_vars = len(qubo_matrix)
        for i in range(num_vars):
            for j in range(num_vars):
                if j < i:
                    continue
                if qubo_matrix[i][j] != 0:
                    qubo[(str(i), str(j))] = float(qubo_matrix[i][j])
        partitioning_encoding = VectorAnnealing.model(qubo, 0)
    return partitioning_encoding


def generate_MQO_encoding_with_matrix(p_queries, p_plan_costs, p_savings_matrix, wl, wm, solver_settings):
    if solver_settings["solver"] == "DA":
        mqo_encoding = MQOQUBOGenerator.generate_Fujitsu_QUBO_with_matrix(p_queries, p_plan_costs, p_savings_matrix, wl, wm)
    elif solver_settings["solver"] == "SA":
        mqo_encoding = MQOQUBOGenerator.generate_DWave_QUBO_with_matrix(p_queries, p_plan_costs, p_savings_matrix, wl, wm)
    elif solver_settings["solver"] == "DWave_Hybrid":
        mqo_encoding = MQOQUBOGenerator.generate_DWave_QUBO_with_matrix(p_queries, p_plan_costs, p_savings_matrix, wl, wm)
    elif solver_settings["solver"] == "SQA":
        mqo_encoding = MQOQUBOGenerator.generate_Qaptiva_QUBO_with_matrix(p_queries, p_plan_costs, p_savings_matrix, wl, wm)
    elif solver_settings["solver"] == "VA":
        mqo_encoding = MQOQUBOGenerator.generate_NEC_QUBO_with_matrix(p_queries, p_plan_costs, p_savings_matrix, wl, wm)
    return mqo_encoding

def solve(encoding, solver_settings, time_limit_sec, data_path, filename):
    if solver_settings["solver"] == "DA":
        import Scripts.FujitsuUtil as FujitsuUtil
        result, opt_time_in_s = FujitsuUtil.solve_problem_v3(encoding, data_path, filename, solver_settings, time_limit_sec, test_with_local_solver=False)
        return result, opt_time_in_s
    elif solver_settings["solver"] == "SA":
        #result, opt_time_in_ms = solve_problem_sa(encoding, solver_settings)
        result, opt_time_in_s = DWaveUtil.solve_problem_SA(encoding, data_path, filename, solver_settings, time_limit_sec)
        return result, opt_time_in_s
    elif solver_settings["solver"] == "DWave_Hybrid":
        result, opt_time_in_s = DWaveUtil.solve_problem_hybrid_bqm(encoding, data_path, filename, solver_settings)
        return result, opt_time_in_s
    elif solver_settings["solver"] == "SQA":
        #result, opt_time_in_ms = solve_problem_sa(encoding, solver_settings)
        import Scripts.QaptivaUtil as QaptivaUtil
        result, opt_time_in_s = QaptivaUtil.solve_problem_SQA(encoding, data_path, filename, solver_settings, time_limit_sec)
        return result, opt_time_in_s
    elif solver_settings["solver"] == "VA":
        import Scripts.NECUtil as NECUtil
        result, opt_time_in_s = NECUtil.solve_problem_VA(encoding, data_path, filename, solver_settings, time_limit_sec)
        return result, opt_time_in_s



def get_query_graph(queries, plan_costs, savings_matrix):
    num_queries = len(queries)
    query_graph = np.zeros((num_queries, num_queries))
    
    for (query1, plans1) in queries.items():
        acc_savings = np.sum(savings_matrix[:, plans1], axis=1)
        for (query2, plans2) in queries.items():
            query_graph[int(query1)][int(query2)] = np.sum(acc_savings[plans2])

    for i in range(num_queries):
        query_graph[i][i] = len(queries[str(i)])
        
    return query_graph

def get_partitioning_QUBO(queries, plan_costs, savings_matrix):
    num_queries = len(queries)
    qubo = np.zeros((num_queries, num_queries))
    query_graph = np.zeros((num_queries, num_queries))
    savings_coeffs = np.zeros((num_queries, num_queries))

    for (query1, plans1) in queries.items():
        acc_savings = np.sum(savings_matrix[:, plans1], axis=1)
        qubo[int(query1)][int(query1)] = qubo[int(query1)][int(query1)] - (pow(len(plans1), 2) * 4)
        query_graph[int(query1)][int(query1)] = len(plans1)
        for (query2, plans2) in queries.items():
            if query2 < query1:
                continue
            if query1 != query2:
                query_graph[int(query1)][int(query2)] = np.sum(acc_savings[plans2])
            
            savings_coeff = np.sum(acc_savings[plans2])
            savings_coeffs[int(query2)][int(query1)] = savings_coeff

            quadr_coeff = 8*len(plans1)*len(plans2) - 2*savings_coeff
            qubo[int(query1)][int(query2)] = qubo[int(query1)][int(query2)] + quadr_coeff
            qubo[int(query2)][int(query1)] = qubo[int(query2)][int(query1)] + quadr_coeff

            lin_coeff = (len(plans1)*len(plans2)*4) - savings_coeff
            qubo[int(query1)][int(query1)] = qubo[int(query1)][int(query1)] - lin_coeff
            qubo[int(query2)][int(query2)] = qubo[int(query2)][int(query2)] - lin_coeff

    return qubo, query_graph, savings_coeffs

def get_split_weight(query_graph, p1, p2):
    split_weight = 0
    for q1 in p1:
        for q2 in p2:
            split_weight = split_weight + query_graph[q1][q2] 
    return split_weight



def parse_query_graph_for_partition(qubo, partition, savings_coeffs, query_graph):
    num_queries = len(partition)
    num_og_queries = len(query_graph)
    org_q_indx_to_p_q_indx = {}
    for i in range(num_queries):
        org_q_indx_to_p_q_indx[partition[i]] = i
    
    p_qubo = np.zeros((num_queries, num_queries))

    for q1 in partition:
        for q2 in partition:
            if q2 < q1:
                continue
            p_idx_1 = org_q_indx_to_p_q_indx[q1]
            p_idx_2 = org_q_indx_to_p_q_indx[q2]
            p_qubo[p_idx_1][p_idx_2] = qubo[q1][q2]
            p_qubo[p_idx_2][p_idx_1] = qubo[q2][q1]
 
            if q1 == q2:
                difference = sum([savings_coeffs[q1][i] - int(query_graph[q1][q1] * query_graph[i][i] *4) for i in range(num_og_queries) if i not in partition])
                p_qubo[p_idx_1][p_idx_1] = p_qubo[p_idx_1][p_idx_1] - difference
    return p_qubo

def process_partitioning_qubo_results(query_graph, qubo_results):
    start_time = time.time()
    bitstrings = []
    min_split_weight = inf
    best_p1 = []
    best_p2 = []
    for solution in qubo_results:
        is_valid = False
        for i in range(int(solution[1])):
            bitstrings.append(solution[0])
            p1 = []
            p2 = []
            for q in range(len(solution[0])):
                b = solution[0][q]
                if b == 0:
                    p1.append(q)
                else:
                    is_valid = True
                    p2.append(q)
            split_weight = get_split_weight(query_graph, p1, p2)
            if is_valid and split_weight < min_split_weight:
                min_split_weight = split_weight
                best_p1 = p1.copy()
                best_p2 = p2.copy()
    total_time_in_s = time.time() - start_time

    return best_p1, best_p2, total_time_in_s

def derive_partitions_multiple(queries, plan_costs, savings_matrix, num_partitioning_phases, solver_settings, data_path):
      
    total_time_in_s = None
    start_time = time.time()
    
    num_queries = len(queries)
    partitions_for_depth = {}
    partitions_for_depth[0] = [np.arange(num_queries).tolist()]

    part_qubo_new_start = time.time()
    part_qubo_matrix, query_graph, savings_coeffs = get_partitioning_QUBO(queries, plan_costs, savings_matrix)
    
    total_time_in_s = time.time() - start_time

    times_for_depth = {}
    for i in range(num_partitioning_phases):
        partitions_for_depth[i+1] = []
        times_for_depth[i] = []
        partition_counter = 0
        for partition in partitions_for_depth[i]:
            p_total_time_in_s = None
            start_time = time.time()
            if i == 0:
                p_part_qubo_matrix = part_qubo_matrix
            else:
                parse_start_time = time.time()
                
                p_part_qubo_matrix = parse_query_graph_for_partition(part_qubo_matrix, partition, savings_coeffs, query_graph)
                
            generate_partitioning_encoding_with_qubo_matrix_start = time.time()
            qubo = generate_partitioning_encoding_with_qubo_matrix(p_part_qubo_matrix, solver_settings)

            p_total_time_in_s = time.time() - start_time

            result, opt_time_in_s = solve(qubo, solver_settings, solver_settings["part_time_limit_sec"], data_path, 'partitioning_response_' + str(i) + '_' + str(partition_counter))
            partition_counter = partition_counter + 1
            
            p_total_time_in_s = p_total_time_in_s + opt_time_in_s
        
            best_p1, best_p2, processing_time_in_s = process_partitioning_qubo_results(query_graph, result)
            p_total_time_in_s = p_total_time_in_s + processing_time_in_s
        
            start_time = time.time()
            best_p1 = [partition[x] for x in best_p1]
            best_p2 = [partition[x] for x in best_p2]

            partitions_for_depth[i+1].append(best_p1)
            partitions_for_depth[i+1].append(best_p2)
           
            p_total_time_in_s = p_total_time_in_s + (time.time() - start_time)
            times_for_depth[i].append(p_total_time_in_s)
        total_time_in_s = total_time_in_s + max(times_for_depth[i])

    return partitions_for_depth[num_partitioning_phases], total_time_in_s

def adjust_partition_plan_costs_for_int_solution(partition, int_solution, plan_costs, savings_matrix, num_plans_per_query):
    start_time = time.time()
    adj_plan_costs = plan_costs.copy()
    for query in partition:
        savings_for_plan = {}
        max_savings = 0
        for p in range(int(query*num_plans_per_query), int(query*num_plans_per_query)+num_plans_per_query):
            savings_for_plan[p] = 0
            for sel_p in int_solution:
                savings_val = savings_matrix[p][sel_p]
                
                if savings_val is not None:
                    savings_for_plan[p] = savings_for_plan[p] + savings_val
            if savings_for_plan[p] > max_savings:
                max_savings = savings_for_plan[p] 
        for p in range(int(query*num_plans_per_query), int(query*num_plans_per_query)+num_plans_per_query):
            adj_plan_costs[p] = adj_plan_costs[p] + max_savings - savings_for_plan[p]
    end_time = time.time()
    return adj_plan_costs

def process_partition(partition, int_solution, queries, plan_costs, savings_matrix, num_plans_per_query, num_og_plans, solver_settings, data_path, partition_idx):
    import time
    total_time_in_s = None
    start_time = time.time()
    if len(int_solution) > 0:
        adjust_start_time = time.time()
        adj_plan_costs = adjust_partition_plan_costs_for_int_solution(partition, int_solution, plan_costs, savings_matrix, num_plans_per_query)
    else:
        adj_plan_costs = plan_costs.copy()
    
    p_queries = {}
    p_plan_costs = []

    p_plans = []
    query_counter = 0
    plan_counter = 0
    p_plan_idx_to_og_plan_ind = {}
    og_plan_idx_to_p_plan_idx = {}
    # Fetch and track partition queries and plan costs
    fetch_partition_queries_start = time.time()
    for q in partition:
        q_plans = queries[str(q)]
        p_queries[str(query_counter)] = []
        for p in q_plans:
            p_plan_costs.append(adj_plan_costs[p])
            p_plans.append(p)
            p_queries[str(query_counter)].append(plan_counter)
            p_plan_idx_to_og_plan_ind[plan_counter] = p
            og_plan_idx_to_p_plan_idx[p] = plan_counter
            plan_counter += 1
        query_counter += 1

    num_p_plans = len(p_plans)
    fetch_partition_savings_start = time.time()

    epsilon = 0.25
    wl = MQOQUBOGenerator.calculate_wl(p_plan_costs, epsilon)

    p_savings_matrix = savings_matrix.copy()
    del_indices = np.arange(num_og_plans)
    del_indices = [x for x in del_indices if x not in p_plans]
    p_savings_matrix = np.delete(p_savings_matrix, del_indices, axis=0)
    p_savings_matrix = np.delete(p_savings_matrix, del_indices, axis=1)
    
    wm = wl + max(np.sum(p_savings_matrix, axis=1))

    p_savings_matrix = np.multiply(p_savings_matrix, -1)

    mqo_encoding_start = time.time()
    
    mqo_encoding = generate_MQO_encoding_with_matrix(p_queries, p_plan_costs, p_savings_matrix, wl, wm, solver_settings)

    total_time_in_s = time.time() - start_time
    
    solutions, opt_time_in_s = solve(mqo_encoding, solver_settings, solver_settings["time_limit_sec"], data_path, "mqo_response")
    total_time_in_s = total_time_in_s + opt_time_in_s
    
    start_time = time.time()
    solution_configurations = []
    for solution in solutions:
        bitstring = solution_to_bitstring(solution, solver_settings)
        p_active_plans = np.argwhere(np.array(bitstring) == 1)
        p_active_plans = [int(x) for x in p_active_plans]
        if not is_solution_valid(p_active_plans, p_queries):
            p_active_plans = postprocess_MQO_solution(p_active_plans, p_plan_costs, p_savings_matrix, len(p_queries.keys()), num_plans_per_query)
        solution_configurations.append(p_active_plans)
    
    total_time_in_s = total_time_in_s + (time.time() - start_time)
    
    return solution_configurations, p_plan_idx_to_og_plan_ind, total_time_in_s

def solution_to_bitstring(solution, solver_settings):
    if solver_settings["solver"] == "DA":
        return solution[0]
    elif solver_settings["solver"] == "SA":
        return solution[0]
    elif solver_settings["solver"] == "DWave_Hybrid":
        return solution[0]
    elif solver_settings["solver"] == "SQA": 
        return solution[0]
    elif solver_settings["solver"] == "VA":
        return solution[0]
    return
    
def postprocess_partition_solutions(partition_solutions, int_solution, partition_to_orginal_labeler, plan_costs, num_plans_per_query, savings_matrix, og_savings, num_og_plans, solver_settings):
    
    start_time = time.time()
    p_min_sample_costs = inf
    p_best_sample_config = None
    for solution in partition_solutions:
        active_plans = [partition_to_orginal_labeler[p] for p in solution]
        
        temp_int_solution = int_solution.copy()
        temp_int_solution.extend(active_plans)
        temp_int_solution = sorted(temp_int_solution)
        costs = evaluate_mqo_solution(temp_int_solution, plan_costs, og_savings, num_og_plans)
        if costs < p_min_sample_costs:
            p_best_sample_config = temp_int_solution
            p_min_sample_costs = costs
          
    end_time = time.time()
    return p_best_sample_config

def get_active_savings(active_plans, savings):

    active_savings = 0
    
    for (p1, p2) in itertools.combinations(active_plans, 2):
        if (p1, p2) in savings:
            active_savings += savings[(p1, p2)]
    
    return active_savings


# In[6]:


def process_single_partition(queries, plan_costs, savings_matrix, og_savings, num_plans_per_query, num_og_plans, solver_settings, data_path):
    total_time_in_s = None
    num_queries = len(queries.keys())
    start_time = time.time()

    epsilon = 0.25
    wl = MQOQUBOGenerator.calculate_wl(plan_costs, epsilon)
    wm = wl + max(np.sum(savings_matrix, axis=1))

    savings_matrix = np.multiply(savings_matrix, -1)
    mqo_encoding = generate_MQO_encoding_with_matrix(queries, plan_costs, savings_matrix, wl, wm, solver_settings)
        
    total_time_in_s = time.time() - start_time
        
    solutions, opt_time_in_s = solve(mqo_encoding, solver_settings, solver_settings["time_limit_sec"], data_path, 'mqo_response')
    total_time_in_s = total_time_in_s + opt_time_in_s
           
    start_time = time.time()
    bitstrings = []

    for solution in solutions:
        for i in range(int(solution[1])):
            bitstrings.append(solution[0])

    min_sample_costs = inf
    best_sample_config = None

    for bitstring in bitstrings:
        active_plans = np.argwhere(np.array(bitstring) == 1)
        active_plans = [int(x) for x in active_plans]
        
        if not is_solution_valid(active_plans, queries):
            active_plans = postprocess_MQO_solution(active_plans, plan_costs, savings_matrix, num_queries, num_plans_per_query)
        
        costs = evaluate_mqo_solution(active_plans, plan_costs, og_savings, num_og_plans)
        if costs < min_sample_costs:
            best_sample_config = active_plans
            min_sample_costs = costs
        
    costs = evaluate_mqo_solution(best_sample_config, plan_costs, og_savings, num_og_plans)
    total_time_in_s = total_time_in_s + (time.time()-start_time)

    return best_sample_config, costs, total_time_in_s

def process_multiple_partitions(queries, plan_costs, savings_matrix, og_savings, num_plans_per_query, num_og_plans, solver_settings, data_path):

    num_partitioning_phases = solver_settings["num_partitioning_phases"]
    partitions, processing_time_in_s = derive_partitions_multiple(queries, plan_costs, savings_matrix, num_partitioning_phases, solver_settings, data_path)
    total_time_in_s = processing_time_in_s

    global_solution = []
    for partition_idx in range(len(partitions)):
        start_time = time.time()
        data_path_part = data_path + '/part_' + str(partition_idx)

        partition = partitions[partition_idx]

        total_time_in_s = total_time_in_s + (time.time() - start_time)
        solutions, p_plan_idx_to_og_plan_ind, processing_time_in_s = process_partition(partition, global_solution, queries, plan_costs, savings_matrix, num_plans_per_query, num_og_plans, solver_settings, data_path_part, partition_idx) 
        total_time_in_s = total_time_in_s + processing_time_in_s
        start_time = time.time()
        
        global_solution = postprocess_partition_solutions(solutions, global_solution, p_plan_idx_to_og_plan_ind, plan_costs, num_plans_per_query, savings_matrix, og_savings, num_og_plans, solver_settings)
        global_solution = sorted(global_solution)
        total_time_in_s = total_time_in_s + (time.time() - start_time)
        
    start_time = time.time()
    global_costs = evaluate_mqo_solution(global_solution, plan_costs, og_savings, num_og_plans)
    total_time_in_s = total_time_in_s + (time.time() - start_time)
    
    return global_solution, global_costs, total_time_in_s
    
def postprocess_MQO_solution(raw_plan_selections, plan_costs, savings, num_queries, num_plans_per_query):
    num_plans = int(num_queries*num_plans_per_query)
    plan_selections_dict = {}
    plan_selections = []
        
    for query in range(num_queries):
        plan_selections_dict[query] = []
    
    for raw_plan_selection in raw_plan_selections:
        query = raw_plan_selection // num_plans_per_query
        plan_selections_dict[query].append(raw_plan_selection)
    
    for (query,plans) in plan_selections_dict.items():
        if len(plans) == 1:
            plan_selections.append(plans[0])
    
    for (query,plans) in plan_selections_dict.items():
        if len(plans) == 1:
            continue
        elif len(plans) == 0:
            candidate_plans = np.arange(int(query*num_plans_per_query), int(query*num_plans_per_query)+num_plans_per_query).tolist()
        else:
            candidate_plans = plans
            
        min_costs = inf
        best_int_config = None
        for candidate_plan in candidate_plans:
            int_plan_selections = plan_selections.copy()
            int_plan_selections.append(candidate_plan)
            costs = evaluate_mqo_solution(int_plan_selections, plan_costs, savings, num_plans)
            if costs < min_costs:
                min_costs = costs
                best_int_config = int_plan_selections
        plan_selections = best_int_config
    
    plan_selections = sorted(plan_selections)
    return plan_selections
       
def conduct_experiment_internal(queries, plan_costs, savings_matrix, num_plans_per_query, num_og_plans, solver_settings, data_path, result_path):
    total_time_in_s = None
    start_time = time.time()
    
    num_queries = len(queries)
    num_plans = len(plan_costs)
    
    og_savings = savings_matrix.copy()
        
    total_time_in_s = time.time() - start_time
    num_partitioning_phases = solver_settings["num_partitioning_phases"]
    if num_partitioning_phases > 0:
        global_plan_configuration, global_costs, processing_time_in_s = process_multiple_partitions(queries, plan_costs, savings_matrix, og_savings, num_plans_per_query, num_og_plans, solver_settings, data_path)
    else:
        global_plan_configuration, global_costs, processing_time_in_s = process_single_partition(queries, plan_costs, savings_matrix, og_savings, num_plans_per_query, num_og_plans, solver_settings, data_path)
    total_time_in_s = total_time_in_s + processing_time_in_s
    
    print("Global configuration:")
    print(global_plan_configuration)
    print("Is global configuration valid:")
    print(is_solution_valid(global_plan_configuration, queries))
    
    global_costs = evaluate_mqo_solution(global_plan_configuration, plan_costs, og_savings, num_og_plans)
    
    print("Total costs: " + str(global_costs) + " obtained after " + str(total_time_in_s) + "s")
    
    export_result(global_plan_configuration, global_costs, total_time_in_s, solver_settings, result_path)


def is_solution_valid(plan_configuration, queries):
    if len(plan_configuration) != len(queries.items()):
        print("Incorrect amount of plans")
        return False
    for i in range(len(plan_configuration)):
        p = plan_configuration[i]
        if p not in queries[str(i)]:
            print("No plan selected for query " + str(i))
            return False
    return True

def get_data_path_for_settings(solver_settings, benchmark, num_queries, num_plans_per_query, problem, data_path_prefix):
    if solver_settings["solver"] == "DA":
        return data_path_prefix + '/DA/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(problem) + '/npart_' + str(solver_settings["num_partitioning_phases"]) + '/time_limit_' + str(solver_settings["time_limit_sec"]) 
    if solver_settings["solver"] == "SA":
        return data_path_prefix + '/SA/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(problem) + '/npart_' + str(solver_settings["num_partitioning_phases"]) + '/' + str(solver_settings["number_iterations"]) + '_iterations' 
    if solver_settings["solver"] == "DWave_Hybrid":
        return data_path_prefix + '/DWave_Hybrid/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(problem) + '/npart_' + str(solver_settings["num_partitioning_phases"]) + '/time_limit_' + str(solver_settings["time_limit"]) 
    if solver_settings["solver"] == "SQA":
        return data_path_prefix + '/SQA/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(problem) + '/npart_' + str(solver_settings["num_partitioning_phases"]) + '/' + str(solver_settings["n_steps"]) + '_iterations'
    if solver_settings["solver"] == "VA":
        return data_path_prefix + '/VA/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(problem) + '/npart_' + str(solver_settings["num_partitioning_phases"]) + '/' + 'vector_mode_' + str(solver_settings["vector_mode"]) + '/' + str(solver_settings["num_sweeps"]) + '_iterations'
    else:
        return ""
        
def conduct_experiments(benchmark_list, num_queries_list, num_plans_per_query_list, problems_list, solver_settings_list, problem_path_prefix, data_path_prefix, result_path_prefix):
     for benchmark in benchmark_list:
        for num_queries in num_queries_list:
            for num_plans_per_query in num_plans_per_query_list:
                num_og_plans = int(num_queries * num_plans_per_query)
                for problem in problems_list:
                    problem_path = problem_path_prefix + '/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/p_' + str(problem)
    
                    queries = DataUtil.load_compressed_data(problem_path, 'queries.txt')
                    plan_costs = DataUtil.load_compressed_data(problem_path, 'plan_costs.txt') 
                    savings_matrix = DataUtil.load_compressed_data(problem_path, 'savings.txt')
                    savings_matrix = np.array(savings_matrix)
                    
                    for solver_settings in solver_settings_list:
                    
                        data_path = get_data_path_for_settings(solver_settings, benchmark, num_queries, num_plans_per_query, problem, data_path_prefix)
                        result_path = result_path_prefix + '/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(problem)
                        if check_if_result_exists(solver_settings, result_path):
                            print("Configuration already processed. Skip.")
                            continue
                                     
                        conduct_experiment_internal(queries.copy(), plan_costs.copy(), savings_matrix.copy(), num_plans_per_query, num_og_plans, solver_settings, data_path, result_path)


def main():
    
    qubo_filename = "qubo_job.json"
    qubo_blob_name = "qubo_job"
    prolog_filename = "prolog.json"
    prolog_blob_name = "prolog"
        
    problem_configurations = [(500, 30, 0, 0.5)]
    problems_list = np.arange(5).tolist()
    benchmark_list = [('job')]
    for problem_configuration in problem_configurations:
        for problem in problems_list:
            problem_path_prefix = 'ExperimentalAnalysis/Problems'
            data_path_prefix = 'ExperimentalAnalysis/Data'
            result_path_prefix = 'ExperimentalAnalysis/Results'
                   
            sample_list = np.arange(1).tolist()

            solver_settings_list = [{'solver': "DA", 'time_limit_sec': 20, 'num_solution': 16, 'num_group': 1, 'timeout': 60, 'num_partitioning_phases': 0, 'qubo_filename': qubo_filename, 'qubo_blob_name': qubo_blob_name, 'prolog_filename': prolog_filename, 'prolog_blob_name': prolog_blob_name},
                                    {'solver': "DA", 'time_limit_sec': 10, 'part_time_limit_sec': 10, 'num_solution': 16, 'num_group': 1, 'timeout': 60, 'num_partitioning_phases': 1, 'qubo_filename': qubo_filename, 'qubo_blob_name': qubo_blob_name, 'prolog_filename': prolog_filename, 'prolog_blob_name': prolog_blob_name},
                                    {'solver': "DA", 'time_limit_sec': 5, 'part_time_limit_sec': 5, 'num_solution': 16, 'num_group': 1, 'timeout': 60, 'num_partitioning_phases': 2, 'qubo_filename': qubo_filename, 'qubo_blob_name': qubo_blob_name, 'prolog_filename': prolog_filename, 'prolog_blob_name': prolog_blob_name}]
    
            conduct_experiments(benchmark_list, [problem_configuration[0]], [problem_configuration[1]], [problem], solver_settings_list, problem_path_prefix, data_path_prefix, result_path_prefix)

# In[ ]:

if __name__ == "__main__":
    main()

