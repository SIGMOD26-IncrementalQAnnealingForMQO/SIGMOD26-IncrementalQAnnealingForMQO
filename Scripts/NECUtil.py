#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import VectorAnnealing
import Scripts.DataUtil as DataUtil
import datetime


# In[ ]:


def parse_solutions_for_serialisation(raw_solutions):
    response = []
    for raw_solution in raw_solutions:
        frequency = 1
        bitstring = [int(x) for x in raw_solution.spin.values()]
        solution = [bitstring, frequency]
        response.append(solution)
    return response

# num_group: Number of independent optimization processes. Increasing the number of independent optimization processes leads to better coverage of the search space. Note: Increasing this number requires to also increase time_limit_sec such that the search time for each process is sufficient. GUI: General / Number optimizations , Default: 1, Min: 1, Max: 16
def solve_problem_VA(qubo, data_path, filename, solver_settings, time_limit, initial_state=None, test_with_local_solver=False):
    
    initial_state=None
    time_limit = 60
    import time
    print("Solve Problem VA")

    sampler = VectorAnnealing.sampler()
    start = time.time()
    print("Vector mode:")
    print(solver_settings["vector_mode"])
    print("Sweeps: " + str(solver_settings["num_sweeps"]))
    print("Reads: " + str(solver_settings["number_runs"]))
    if initial_state is not None:
        raw_results = sampler.sample(qubo, vector_mode=solver_settings["vector_mode"], num_sweeps=solver_settings["num_sweeps"], num_reads=solver_settings["number_runs"], init_spin=initial_state)
    else:
        raw_results = sampler.sample(qubo, vector_mode=solver_settings["vector_mode"], num_sweeps=solver_settings["num_sweeps"], num_reads=solver_settings["number_runs"])

    results = parse_solutions_for_serialisation(raw_results)
    
    opt_time = time.time() - start
    
    print("Opt time: " + str(opt_time))
    
    data = {}
    data["solutions"] = results
    data["execution_time"] = opt_time
    
    print("Data path:")
    print(data_path)
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")
    
    return results, opt_time

