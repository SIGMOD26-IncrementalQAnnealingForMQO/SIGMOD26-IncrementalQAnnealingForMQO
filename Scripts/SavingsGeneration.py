#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import os
import pickle
import pathlib
import itertools
import Scripts.DataUtil as DataUtil


def sample_uniform(min_val=1, max_val=10, size=1, scalar=1):
    return np.random.randint(min_val, max_val, size=size, dtype=int) * scalar

def generate_MQO_problem(num_queries, num_plans_per_query, num_communities, qubo_density_in_min, qubo_density_in_max, qubo_density_out, cost_domain_bound=20, savings_domain_bound=10, enforce_identical_community_sizes=True):
    num_plans = num_queries * num_plans_per_query
    
    queries = {}

    for q in range(num_queries):
        queries[q] = []
        for p in range(num_plans_per_query):
            queries[q].append(q*num_plans_per_query + p)

    savings, densities = generate_savings_internal(queries, num_plans_per_query, num_communities, qubo_density_in_min, qubo_density_in_max, qubo_density_out, savings_domain_bound, enforce_identical_community_sizes=enforce_identical_community_sizes)
    
    return savings

def generate_savings_internal(queries, num_plans_per_query, num_communities, qubo_density_in_min, qubo_density_in_max, qubo_density_out, savings_domain_bound, enforce_identical_community_sizes=True):

    num_queries = len(queries.keys())
    num_plans = num_queries * num_plans_per_query
    
    savings = np.zeros((num_plans, num_plans), dtype=int).tolist()
    
    query_indices = np.arange(num_queries)
    if enforce_identical_community_sizes:
        sample_size = int(num_queries / num_communities)
        communities = {}
        for i in range(num_communities):
            community = np.random.choice(query_indices, sample_size, replace=False)
            community = sorted(community)
            communities[i] = community
            query_indices = [x for x in query_indices if x not in community]
            
        remainder_counter = 0
        while len(query_indices) > 0:
            if remainder_counter == num_communities:
                remainder_counter = 0
            remainder_query = query_indices.pop(0)
            communities[remainder_counter].append(remainder_query)
            remainder_counter = remainder_counter + 1    
    else:
        communities = {}
        for i in range(num_communities):
            communities[i] = []
        while len([x for x in communities.values() if len(x) < 2]) > 0:
            split_points = np.random.choice(len(query_indices)-1, num_communities-1, replace=False) + 1
            split_points.sort()
            raw_communities = np.split(query_indices, split_points)
            communities = {}
            for i in range(len(raw_communities)):
                communities[i] = raw_communities[i].tolist()

    
    community_for_plan = {}
    sampled_densities = {}
    for i in range(num_communities):
        
        plan_set = []
        for query_index in communities[i]:
            query_plans = queries[query_index]
            plan_set.extend(query_plans)
            for query_plan in query_plans:
                community_for_plan[query_plan] = i
        
        savings_candidates = list(x for x in itertools.combinations(plan_set, 2) if int(x[0]/num_plans_per_query) != int(x[1]/num_plans_per_query))
       
        total_num_savings = len(savings_candidates)
        qubo_density_in = np.random.uniform(low=qubo_density_in_min, high=qubo_density_in_max, size=1)[0]
        sampled_densities[i] = {'min': qubo_density_in_min, 'max': qubo_density_in_max, 'sampled': qubo_density_in}
        num_savings = int(total_num_savings*qubo_density_in)
        
        
        if num_savings >= 1:
            plan_pair_indices = np.random.choice(len(savings_candidates), num_savings, replace=False)
            plan_pairs = np.array(savings_candidates)[plan_pair_indices]
            savings_vals = sample_uniform(size=num_savings, max_val=savings_domain_bound)
            for i in range(len(plan_pairs)):
                savings[int(plan_pairs[i][0])][int(plan_pairs[i][1])] = int(savings_vals[i])
                savings[int(plan_pairs[i][1])][int(plan_pairs[i][0])] = int(savings_vals[i])
        else:
            print("Density too small - no savings")
            return 
    
    savings_candidates = list(x for x in itertools.combinations(np.arange(num_plans), 2) if community_for_plan[x[0]] != community_for_plan[x[1]])
    total_num_savings = len(savings_candidates)
    inter_community_density = qubo_density_out
    num_savings = int(total_num_savings*inter_community_density)
    
    if num_savings >= 1:
        plan_pair_indices = np.random.choice(len(savings_candidates), num_savings, replace=False)
        
        plan_pairs = np.array(savings_candidates)[plan_pair_indices]
        savings_vals = sample_uniform(size=num_savings, max_val=savings_domain_bound)
        
        for i in range(len(plan_pairs)):
            savings[int(plan_pairs[i][0])][int(plan_pairs[i][1])] = int(savings_vals[i])
            savings[int(plan_pairs[i][1])][int(plan_pairs[i][0])] = int(savings_vals[i])

    return savings, sampled_densities

   
def generate(problem_configurations, problems_list, problem_path_prefix):
    import hashlib
    for (num_queries, num_plans_per_query, num_communities, enforce_identical_community_sizes, density_in_min, density_in_max, density_out, cost_domain_bound, savings_domain_bound) in problem_configurations:
        for problem in problems_list:
            if enforce_identical_community_sizes:
                problem_path = problem_path_prefix + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/' + str(num_communities) + '_c/d_in_min_' + str(density_in_min) + '/d_in_max_' + str(density_in_max) + '/d_out_' + str(density_out) + '/cd_' + str(cost_domain_bound) + '/sd_' + str(savings_domain_bound) + '/p_' + str(problem)
            else:
                problem_path = problem_path_prefix + '/' + str(num_queries) + '_q/' + str(num_plans_per_query) + '_ppq/' + str(num_communities) + '_c_us/d_in_min_' + str(density_in_min) + '/d_in_max_' + str(density_in_max) + '/d_out_' + str(density_out) + '/cd_' + str(cost_domain_bound) + '/sd_' + str(savings_domain_bound) + '/p_' + str(problem)

            h = hashlib.new('sha256')
            h.update(problem_path.encode())
            seed = np.frombuffer(h.digest(), dtype='uint32')

            np.random.seed(seed)
            
            savings = generate_savings(num_queries, num_plans_per_query, num_communities, density_in_min, density_in_max, density_out, cost_domain_bound=cost_domain_bound, savings_domain_bound=savings_domain_bound, enforce_identical_community_sizes=enforce_identical_community_sizes)
            
            DataUtil.compress_and_save_data(savings, problem_path, 'savings.txt')

