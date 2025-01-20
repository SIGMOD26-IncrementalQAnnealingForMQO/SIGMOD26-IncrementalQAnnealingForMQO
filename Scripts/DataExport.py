#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import os
import pickle
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
        
def export_temporal_cost_evolution_to_csv(benchmarks, benchmark_labels, num_queries_list, num_plans_per_query_list, problems_list, algorithms_list, time_intervals, problem_path_prefix, result_path_prefix, result_filename, include_header=True, na_costs=3, timeout_in_ms=300000):
    csv_data_list = []
    if include_header:
        csv_data_list.append(["algorithm", "benchmark", "num_queries", "num_plans_per_query", "problem_index", "opt_time", "costs", "normalised_costs"])
    
    for benchmark in benchmarks:
        for num_queries in num_queries_list:
            for num_plans_per_query in num_plans_per_query_list:
                    #for cost_scaling in cost_scaling_list:
                for p in problems_list:
                    problem_path = problem_path_prefix + '/CS0/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/p_' + str(p)
                    
                    if not os.path.exists(problem_path):
                        continue
                    print(problem_path)
                    
                            
                    min_costs = inf
                    problem_results_list = []
                    
                    for algorithm in algorithms_list:
                        result_path = result_path_prefix + '/' + benchmark + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/prob_' + str(p)
                        result = load_data(result_path, algorithm + '.json')
                        
                        for time_interval in time_intervals:
                            if result is None or len(result) == 0 or float(result[0]["time_in_ms"]) > time_interval :
                                problem_results_list.append([algorithm, benchmark_labels[benchmark], num_queries, num_plans_per_query, p, time_interval, "n/a", na_costs])
                                continue
                            result_index = 0
                            result_index_time = float(result[result_index]["time_in_ms"])

                            while result_index < len(result)-1 and float(result[result_index+1]["time_in_ms"]) <= time_interval:
                                result_index = result_index + 1
                                result_index_time = float(result[result_index]["time_in_ms"])

                            result_for_interval = result[result_index]
                            costs = result_for_interval["costs"]
                            
                            if costs < min_costs:
                                min_costs = costs
                            problem_results_list.append([algorithm, benchmark_labels[benchmark], num_queries, num_plans_per_query, p, time_interval, costs, na_costs])

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