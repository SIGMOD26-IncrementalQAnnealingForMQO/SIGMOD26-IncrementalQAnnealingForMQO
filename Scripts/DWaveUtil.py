#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dwave.system import LeapHybridSampler
from dwave.system import LeapHybridDQMSampler
import Scripts.DataUtil as DataUtil
from dwave.cloud.client import Client

import neal
import time
import numpy as np


# In[ ]:


def response_to_dict(raw_response, use_classical_solver=True):
    response = []
    for i in range(len(raw_response.record)):
        if use_classical_solver:
            (sample, energy, occ) = raw_response.record[i]
            response.append([sample.tolist(), occ.item(), energy.item()])
        else:
            (sample, energy, occ, chain) = raw_response.record[i]
            response.append([sample.tolist(), occ.item(), energy.item()])
    return response

def precise_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

# In[ ]:

def get_minimum_time(num_variables, time_progression):
    #time_progression = [[1,3],[1024,3],[4096,10],[10000,40],[30000,200],[100000,600],[1000000,600]]
    
    step = 0
    upper_num_variables = time_progression[step][0]
    while upper_num_variables < num_variables:
        step = step + 1
        upper_num_variables = time_progression[step][0]
        
    lower_time = time_progression[step-1][1]
    lower_num_variables = time_progression[step-1][0]
    upper_time = time_progression[step][1]
    
    progression_rate = (num_variables-lower_num_variables) / (upper_num_variables-lower_num_variables)
    min_time_limit = progression_rate*(upper_time-lower_time) + lower_time
    return precise_floor(min_time_limit, precision=2) + 0.01

# 'minimum_time_limit': [[1, 3.0], [1024, 3.0], [4096, 10.0], [10000, 40.0], [30000, 200.0], [100000, 600.0], [1000000, 600.0]]
def solve_problem_hybrid_bqm(bqm, data_path, filename, solver_settings):
    client = Client.from_config(config_file='license/dwave.conf', profile='default')
    
    print("Solve Problem D-Wave Hybrid")

    # TODO!
    #print(LeapHybridSampler().min_time_limit(bqm))
    #print(dir(LeapHybridSampler()))
    #print(LeapHybridSampler().parameters)
    #print(print(LeapHybridSampler().properties))
    solver = client.get_solver('hybrid_binary_quadratic_model_version2p')
    
    #print(dir(solver))
    #print(solver.properties)
    #print(solver.check_problem(bqm))
        
    #raw_response = LeapHybridSampler().sample(bqm, time_limit=time_limit) 
    
    time_progression = solver.properties['minimum_time_limit']
    num_variables = bqm.num_variables
    
    min_time = get_minimum_time(num_variables, time_progression)
    
    print("Min time: " + str(min_time))
    
    if min_time > 300:
        print("Processing time too high. Abort")
        return
    
   
    #raw_response = solver.sample_bqm(bqm, time_limit=time_limit).result()["sampleset"]
    raw_response = solver.sample_bqm(bqm, time_limit=min_time).result()["sampleset"]
    
    info = raw_response.info
    print(raw_response)
    print(raw_response.info)
    
    response = response_to_dict(raw_response, use_classical_solver=True)
    print(response)
    
    data = {}
    data["solutions"] = response
    for (k,v) in info.items():
        data[k] = v
    print(data)
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")

    run_time_in_μs = info["run_time"]
    run_time_in_s = info["run_time"] / 1000000
    return response, run_time_in_s
    
def solve_problem_hybrid_bqm_old(bqm, data_path, filename, time_limit):
    client = Client.from_config(config_file='license/dwave.conf', profile='default')
    
    print("Solve Problem D-Wave Hybrid")

    # TODO!
    #print(LeapHybridSampler().min_time_limit(bqm))
    #print(dir(LeapHybridSampler()))
    #print(LeapHybridSampler().parameters)
    #print(print(LeapHybridSampler().properties))
    solver = client.get_solver('hybrid_binary_quadratic_model_version2p')
    print(dir(solver))
    #print(dir(solver))
    #raw_response = LeapHybridSampler().sample(bqm, time_limit=time_limit) 
    raw_response = solver.sample_bqm(bqm, time_limit=time_limit).result()
    print("raw_response")
    print(raw_response)
    print("dir(raw_response)")
    print(dir(raw_response))
    for raw_response_element in raw_response:
        print("raw_response_element")
        print(raw_response_element)
        print("dir(raw_response_element)")
        print(dir(raw_response_element))
    #raw_response = LeapHybridSampler().sample(bqm) 
    
    info = raw_response.info
    print(raw_response)
    print(raw_response.info)
    
    response = response_to_dict(raw_response, use_classical_solver=True)
    print(response)
    
    data = {}
    data["solutions"] = response
    for (k,v) in info.items():
        data[k] = v
    print(data)
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")

    run_time_in_μs = info["run_time"]
    run_time_in_s = info["run_time"] / 1000000
    return response, run_time_in_s
    
# TODO
#def solve_problem_hybrid_dqm(dqm, time_limit=10, number_runs=100):
def solve_problem_SA(bqm, data_path, filename, solver_settings, time_limit):
    number_runs= solver_settings["number_runs"]
    number_iterations= solver_settings["number_iterations"]

    sampler = neal.SimulatedAnnealingSampler()
    
    start = time.time()
    result = sampler.sample(bqm, num_reads=number_runs, num_sweeps=number_iterations, answer_mode='raw', time_limit=time_limit)
    opt_time = time.time() - start
    data = {}
    
    solutions = []
    for item in result.record:
        bitstring = [int(x) for x in item[0]]
        solutions.append([bitstring, int(item[2]), float(item[1])])
        print("Energy: " + str(float(item[1])))
        
    data["solutions"] = solutions
    data["execution_time"] = opt_time
    
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")
    return solutions, opt_time


def test_time():
    client = Client.from_config(config_file='license/dwave.conf', profile='default')
    solver = client.get_solver('hybrid_binary_quadratic_model_version2p')
    
    time_progression = solver.properties['minimum_time_limit']
    num_queries = 500
    num_ppq = 40
    num_variables = int(num_queries*num_ppq)
 
    
    min_time = get_minimum_time(num_variables, time_progression)
    print("Min time: " + str(min_time))