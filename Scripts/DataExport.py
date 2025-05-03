#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import os
import pathlib
import itertools

import time
from math import inf
import csv

import Scripts.DataUtil as DataUtil


# In[2]:


def save_to_csv(data, path, filename):
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()


def load_data(path, filename):
    datafile = os.path.abspath(path + '/' + filename)
    if os.path.exists(datafile):
        with open(datafile, 'rb') as file:
            return json.load(file)
        
def load_all_results(path):
    if not os.path.isdir(path):
        return []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    for datafile in onlyfiles:
        with open(path + '/' + datafile, 'rb') as file:
            data.append(json.load(file))
    return data

def save_data(data, path, filename):
    print(path)
    datapath = os.path.abspath(path)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path + '/' + filename)
    mode = 'a' if os.path.exists(datafile) else 'w'
    with open(datafile, mode) as file:
        json.dump(data, file)


# In[3]:


def translate_baseline_solution(num_queries, num_plans_per_query, solution):
    translated_solution = []
    for query in range(num_queries):
        translated_solution.append(query*num_plans_per_query+solution[query])
    return translated_solution

def export_baseline_solutions(benchmarks, num_queries_list, num_plans_per_query_list, problems_list, algorithms_list, problem_path_prefix, baseline_path_prefix, result_path_prefix):
    
    for benchmark in benchmarks:
        for num_queries in num_queries_list:
            for num_plans_per_query in num_plans_per_query_list:
                for p in problems_list:
                    problem_path = problem_path_prefix + '/CS0/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/p_' + str(p)
                    if not os.path.exists(problem_path):
                        continue
                    for algorithm in algorithms_list:
                        baseline_path = baseline_path_prefix + '/CS0/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/p_' + str(p)
                        if not os.path.exists(baseline_path):
                            continue
                        baseline_result = load_data(baseline_path, algorithm + '.json')
                            
                        if baseline_result is None:
                            continue

                        result = []

                        min_costs = inf
                        min_time_interval = None
                        if baseline_result["costs"] != "n/a":
                            for (time_interval, costs) in baseline_result["cost_evolution"].items():
                                if costs != "Infinity" and int(costs) < min_costs:
                                    if int(costs) < 100:
                                        print(baseline_path)
                                        print(int(costs))
                                    min_costs = costs
                                    min_time_interval = time_interval
                                    result.append({"time_in_ms": time_interval, "costs": costs, "plan_selection": "n/a"})
                        if min_time_interval != None:
                            result[-1]["plan_selection"] = translate_baseline_solution(num_queries, num_plans_per_query, baseline_result["plan_selection"])
                        result_path = result_path_prefix + '/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(p)
                        DataUtil.save_data(result, result_path, algorithm + '.json', override=True)

def export_cost_results_to_csv(num_queries_list, num_plans_per_query_list, density_list, cost_scaling_list, problems_list, algorithms_list, result_path_prefix, include_header=True, na_costs=3, timeout_in_ms=60000):
    csv_data_list = []
    if include_header:
        csv_data_list.append(["algorithm", "num_queries", "num_plans_per_query", "density", "problem_index", "costs", "normalised_costs"])

    for benchmark in benchmarks:
        for num_queries in num_queries_list:
            for num_plans_per_query in num_plans_per_query_list:
                for p in problems_list:
                    min_costs = inf
                    problem_results_list = []
                    for algorithm in algorithms_list:
                        result_path = result_path_prefix + '/' + str(num_queries) + '_queries/' + str(num_plans_per_query) + '_ppq/density_' + str(density) + '/cost_scaling_' + str(cost_scaling) + '/problem_' + str(p)
                        result = load_data(result_path, algorithm + '.json')
                        if result is None or len(result) == 0 or float(result[0]["time_in_ms"]) > timeout_in_ms:
                            problem_results_list.append([algorithm, num_queries, num_plans_per_query, density, p, "n/a", na_costs])
                            continue
                        result_index = 0
                        result_index_time = float(result[result_index]["time_in_ms"])
                        while result_index < len(result)-1 and float(result[result_index+1]["time_in_ms"]) <= timeout_in_ms:
                            result_index = result_index + 1
                            result_index_time = float(result[result_index]["time_in_ms"])

                        result = result[result_index]
                        costs = result["costs"]
                        if costs < min_costs:
                            min_costs = costs
                        problem_results_list.append([algorithm, num_queries, num_plans_per_query, density, p, costs, na_costs])

                    for problem_result in problem_results_list:
                        problem_costs = problem_result[-2]
                        if problem_costs != "n/a":
                            normalised_costs = problem_costs / min_costs
                                
                            if normalised_costs > na_costs:
                                normalised_costs = na_costs
                            problem_result[-1] = normalised_costs
                        csv_data_list.append(problem_result.copy())

    for csv_data in csv_data_list:
        save_to_csv(csv_data, result_path_prefix, 'results.txt')
 
 
def get_size_equal_string(size_equal):
    if size_equal == "es":
        return "true"
    else:
        return "false"     
        
def export_csv_costs(num_queries_list, num_plans_per_query_list, community_configuration_list, size_equal_configs, density_in_list, density_out_list, cost_domain_bound_list, savings_domain_bound_list, problems_list, algorithms_list, problem_path_prefix, result_path_prefix, result_filename, include_header=True, na_costs=3):
    csv_data_list = []
    if include_header:
        csv_data_list.append(["algorithm", "num_queries", "num_plans_per_query", "num_communities", "equal_sizes", "density_in_min", "density_in_max", "density_out", "cost_domain_bound", "savings_domain_bound", "problem_index", "opt_time", "costs", "normalised_costs"])
    
    for num_queries in num_queries_list:
        for num_plans_per_query in num_plans_per_query_list:
            num_og_plans = int(num_queries * num_plans_per_query)
            for num_communities in community_configuration_list:
                for (density_in_min, density_in_max) in density_in_list:
                    for density_out in density_out_list:
                        for cost_domain_bound in cost_domain_bound_list:
                            for size_equal in size_equal_configs:
                                for savings_domain_bound in savings_domain_bound_list:
                                    for problem in problems_list:
                                        if size_equal == "es": # equal sizes
                                            problem_path = problem_path_prefix + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/' + str(num_communities) + '_c/d_in_min_' + str(density_in_min) + '/d_in_max_' + str(density_in_max) + '/d_out_' + str(density_out) + '/cd_' + str(cost_domain_bound) + '/sd_' + str(savings_domain_bound) + '/p_' + str(problem)
                                        else: # unequal sizes
                                            problem_path = problem_path_prefix + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/' + str(num_communities) + '_c_us/d_in_min_' + str(density_in_min) + '/d_in_max_' + str(density_in_max) + '/d_out_' + str(density_out) + '/cd_' + str(cost_domain_bound) + '/sd_' + str(savings_domain_bound) + '/p_' + str(problem)

                                        if not os.path.exists(problem_path):
                                            continue
                                        
                                        min_costs = inf
                                        problem_results_list = []
                                        
                                        da_results_missing = False
                                        for algorithm in algorithms_list:
                                            if algorithm == "HQA_incremental" and problem > 0:
                                                continue
                                            if algorithm == "HQA_incremental" and num_queries > 500:
                                                continue    
                                            result_path = result_path_prefix + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/' + str(num_communities) + '_c/' + size_equal + '/d_in_min_' + str(density_in_min) + '/d_in_max_' + str(density_in_max) + '/d_out_' + str(density_out) + '/cd_' + str(cost_domain_bound) + '/sd_' + str(savings_domain_bound) + '/p_' + str(problem)
                                            result = load_data(result_path, algorithm + '.json')
                                            if result is None and algorithm == 'incremental':
                                                da_results_missing = True
                                  
                                            if result is None or len(result) == 0:
                                                problem_results_list.append([algorithm, num_queries, num_plans_per_query, num_communities, get_size_equal_string(size_equal), density_in_min, density_in_max, density_out, cost_domain_bound, savings_domain_bound, problem, 0, "n/a", na_costs])
                                                continue
                                                
                                            opt_time = result[-1]["time_in_ms"]
                                            costs = result[-1]["costs"]
                                                
                                            if costs < min_costs:
                                                min_costs = costs
                                            problem_results_list.append([algorithm, num_queries, num_plans_per_query, num_communities, get_size_equal_string(size_equal), density_in_min, density_in_max, density_out, cost_domain_bound, savings_domain_bound, problem, opt_time, costs, na_costs])
                                                
                                                
                                        if da_results_missing:
                                            continue
                                        for problem_result in problem_results_list:
                                            problem_costs = problem_result[-2]
                                            if problem_costs != "n/a":
                                                normalised_costs = problem_costs / min_costs
                                                 
                                                if normalised_costs > na_costs:
                                                    normalised_costs = na_costs
                                                problem_result[-1] = normalised_costs
                                            csv_data_list.append(problem_result.copy())
    
    for csv_data in csv_data_list:
        save_to_csv(csv_data, result_path_prefix, result_filename)
   
   
def export_benchmark_results_to_csv(benchmarks, num_queries_list, num_plans_per_query_list, problems_list, algorithms_list, problem_path_prefix, result_path_prefix, result_filename, include_header=True, na_costs=3, timeout_in_ms=300000):
    csv_data_list = []
    if include_header:
        csv_data_list.append(["algorithm", "benchmark", "num_queries", "num_plans_per_query", "problem_index", "costs", "normalised_costs"])
    
    for benchmark in benchmarks:
        for num_queries in num_queries_list:
            for num_plans_per_query in num_plans_per_query_list:
                for p in problems_list:
                    problem_path = problem_path_prefix + '/CS0/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/p_' + str(p)
                    print(problem_path)
                    if not os.path.exists(problem_path):
                        continue
                            
                    min_costs = inf
                    problem_results_list = []
                    all_results_collected = True
                    for algorithm in algorithms_list:
                        result_path = result_path_prefix + '/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(p)
                        print(result_path)
                        result = load_data(result_path, algorithm + '.json')
                        if result is None:
                            all_results_collected = False
                            continue
               
                        costs = result["costs"]
                          
                        if costs < min_costs:
                            min_costs = costs
                        problem_results_list.append([algorithm, benchmark, num_queries, num_plans_per_query, p, costs, na_costs])

                    if not all_results_collected:
                        continue
                    
                    for problem_result in problem_results_list:
                        problem_costs = problem_result[-2]
                        if problem_costs != "n/a":
                            normalised_costs = problem_costs / min_costs
                             
                            if normalised_costs > na_costs:
                                normalised_costs = na_costs
                            problem_result[-1] = normalised_costs
                        csv_data_list.append(problem_result.copy())
    
    for csv_data in csv_data_list:
        save_to_csv(csv_data, result_path_prefix, result_filename)
  
