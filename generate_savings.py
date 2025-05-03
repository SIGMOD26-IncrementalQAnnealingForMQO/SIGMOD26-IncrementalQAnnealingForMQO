#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import os
import pathlib
import itertools

import Scripts.MQOQUBOGenerator as MQOQUBOGenerator

import Scripts.DataExport as DataExport
import Scripts.DataUtil as DataUtil
import Scripts.SavingsGeneration as SavingsGeneration
import time
from math import inf
   

def generate_savings(num_queries_list, num_plans_per_query_list, community_configuration_list, density_in_list, density_out_list, cost_domain_bound_list, savings_domain_bound_list, problems_list, problem_path_prefix):
    for num_queries in num_queries_list:
        for num_plans_per_query in num_plans_per_query_list:
            num_og_plans = int(num_queries * num_plans_per_query)
            for num_communities in community_configuration_list:
                for (density_in_min, density_in_max) in density_in_list:
                    for density_out in density_out_list:
                        for cost_domain_bound in cost_domain_bound_list:
                            for savings_domain_bound in savings_domain_bound_list:
                                for problem in problems_list:
                                    problem_configurations = [(num_queries, num_plans_per_query, num_communities, False, density_in_min, density_in_max, density_out, cost_domain_bound, savings_domain_bound)]
                                    SavingsGeneration.generate(problem_configurations, [problem], problem_path_prefix)


def main():
    
    problem_path_prefix = 'ExperimentalAnalysis/CommunityProblems'
    
    
    num_queries_list = [250, 500, 750, 1000]
    num_plans_per_query_list = [20, 30, 40]
    community_configurations_list = [4]

    density_in_list = [(0.05, 1)] 
    density_out_list = [0.05]
    cost_domain_bound_list = [20]
    savings_domain_bound_list = [10]
    problems_list = [0, 1, 2]

    generate_savings(num_queries_list, num_plans_per_query_list, community_configurations_list, density_in_list, density_out_list, cost_domain_bound_list, savings_domain_bound_list, problems_list, problem_path_prefix)

    num_queries_list = [250, 500, 750, 1000]
    num_plans_per_query_list = [30]
    community_configurations_list = [1, 2, 6, 10]

    density_in_list = [(0.05, 1)] 
    density_out_list = [0.05]
    cost_domain_bound_list = [20]
    savings_domain_bound_list = [10]
    problems_list = [0, 1, 2]
    
    generate_savings(num_queries_list, num_plans_per_query_list, community_configurations_list, density_in_list, density_out_list, cost_domain_bound_list, savings_domain_bound_list, problems_list, problem_path_prefix)

    num_queries_list = [250, 500, 750, 1000]
    num_plans_per_query_list = [30]
    community_configurations_list = [4]

    density_in_list = [(0.05, 0.25), (0.05, 0.5), (0.05, 0.75)] 
    density_out_list = [0.05]
    cost_domain_bound_list = [20]
    savings_domain_bound_list = [10]
    problems_list = [0, 1, 2]
    
    generate_savings(num_queries_list, num_plans_per_query_list, community_configurations_list, density_in_list, density_out_list, cost_domain_bound_list, savings_domain_bound_list, problems_list, problem_path_prefix)



if __name__ == "__main__":
    main()


