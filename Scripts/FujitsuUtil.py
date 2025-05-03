from dadk.QUBOSolverDAv3c import QUBOSolverDAv3c
from dadk.QUBOSolverCPU import *
import Scripts.DataUtil as DataUtil
import datetime


def parse_solutions_for_serialisation(raw_solutions):
    response = []
    for raw_solution in raw_solutions:
        solution = [raw_solution.configuration, int(raw_solution.frequency), float(raw_solution.energy)]
        response.append(solution)
    return response

def solve_problem_v3(fujitsu_qubo, data_path, filename, solver_settings, time_limit, test_with_local_solver=False):
    if test_with_local_solver:
        solver = QUBOSolverCPU(number_runs=solver_settings["num_solution"])
    else:
        solver = QUBOSolverDAv3c(time_limit, timeout=solver_settings["timeout"], num_solution=solver_settings["num_solution"], num_output_solution=solver_settings["num_solution"], num_group=solver_settings["num_group"], access_profile_file='annealer.prf', use_access_profile=True, qubo_blob_name=solver_settings['qubo_blob_name'], prolog_filename=solver_settings['prolog_filename'], prolog_blob_name=solver_settings['prolog_blob_name'])

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
        #if info["label"] == "Solution mode":
            #continue
        if info["label"] == "Execution time":
            execution_time = info["value"].total_seconds()
        if isinstance(info["value"], datetime.timedelta):
            data[info["label"]] = info["value"].total_seconds()
        else:
            data[info["label"]] = str(info["value"])
    DataUtil.compress_and_save_data(data, data_path, filename + ".txt")
    
    return result, execution_time

