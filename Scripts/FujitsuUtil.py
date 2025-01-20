#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dadk.QUBOSolverDAv3c import QUBOSolverDAv3c
from dadk.QUBOSolverCPU import *
import Scripts.DataUtil as DataUtil
import datetime


# In[ ]:


def parse_solutions_for_serialisation(raw_solutions):
    response = []
    for raw_solution in raw_solutions:
        solution = [raw_solution.configuration, int(raw_solution.frequency), float(raw_solution.energy)]
        response.append(solution)
    return response

# num_group: Number of independent optimization processes. Increasing the number of independent optimization processes leads to better coverage of the search space. Note: Increasing this number requires to also increase time_limit_sec such that the search time for each process is sufficient. GUI: General / Number optimizations , Default: 1, Min: 1, Max: 16
def solve_problem_v3(fujitsu_qubo, data_path, filename, solver_settings, time_limit, test_with_local_solver=False):
    print("Solve Problem DA")
    if test_with_local_solver:
        solver = QUBOSolverCPU(number_runs=number_runs)
    else:
        solver = QUBOSolverDAv3c(time_limit, timeout=solver_settings["timeout"], num_solution=solver_settings["num_solution"], num_output_solution=solver_settings["num_solution"], num_group=solver_settings["num_group"], access_profile_file='annealer.prf', offline_request_file='da_request.txt', use_access_profile=True, qubo_filename=solver_settings['qubo_filename'], qubo_blob_name=solver_settings['qubo_blob_name'], prolog_filename=solver_settings['prolog_filename'], prolog_blob_name=solver_settings['prolog_blob_name'])

    fail_counter = 0
    while True:
        if fail_counter >= 3:
            print("Maximum failure rate exceeded")
            print("Abort")
            return
        try:
            solution_list = solver.minimize(fujitsu_qubo)
            break
        except Exception:
            traceback.print_exc()
            print("Library error. Repeating request")
            fail_counter = fail_counter + 1
    
    solutions = solution_list.solutions
    
    result = parse_solutions_for_serialisation(solutions)
    data = {}
    data["solutions"] = result
    execution_time = None
    for info in solution_list.stats_info:
        if info["label"] == "Execution time":
            execution_time = info["value"].total_seconds()
        if isinstance(info["value"], datetime.timedelta):
            data[info["label"]] = info["value"].total_seconds()
        else:
            data[info["label"]] = info["value"]
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")
    
    return result, execution_time

