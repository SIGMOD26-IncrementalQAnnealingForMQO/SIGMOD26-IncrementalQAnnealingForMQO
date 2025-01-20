#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dwave.system import LeapHybridSampler
from dwave.system import LeapHybridDQMSampler
import Scripts.DataUtil as DataUtil
from dwave.cloud.client import Client

import neal
import time


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



# 'minimum_time_limit': [[1, 3.0], [1024, 3.0], [4096, 10.0], [10000, 40.0], [30000, 200.0], [100000, 600.0], [1000000, 600.0]]
def solve_problem_hybrid_bqm(bqm, data_path, filename, time_limit):
    client = Client.from_config(config_file='license/dwave.conf', profile='default')
    print("Solve Problem D-Wave Hybrid")

    raw_response = LeapHybridSampler().sample(bqm, time_limit=time_limit) 
    
    info = raw_response.info
    
    response = response_to_dict(raw_response, use_classical_solver=True)
    
    data = {}
    data["solutions"] = response
    for (k,v) in info.items():
        data[k] = v
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")

    run_time_in_μs = info["run_time"]
    run_time_in_s = info["run_time"] / 1000000
    return response, run_time_in_s
    
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
        
    data["solutions"] = solutions
    data["execution_time"] = opt_time
    
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")
    return solutions, opt_time
