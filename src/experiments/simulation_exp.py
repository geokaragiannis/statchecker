"""
Includes several simulation experiments
"""
DATA_PATH = "data/claims_02-19-2020/"

import pandas as pd
from src.pipeline.classification_step import ClassificationStep
from src.pipeline.optimization_step import OptimizationStep
from src.crowdsourcing.simulation import Simulation
import json

def get_stats(claims, stats_dict):
    for claim in claims:
        for prop in claim.available_properties:
            stats_dict[prop.property_name][prop.verified_index] += 1
            stats_dict["all_props"][prop.verified_index] += 1
    print(json.dumps(stats_dict))


class_step = ClassificationStep(DATA_PATH, simulation=False, export=True)
complete_df = class_step.parser.get_complete_df()

# create a test dataframe
test_df = complete_df.sample(n=100, random_state=2)

# remove the claims selected for testing
complete_df = test_df.merge(complete_df, how = 'outer', indicator=True).loc[lambda x : x['_merge']=='right_only']
complete_df.drop(columns=["_merge"], inplace=True)
# train
# class_step.train_for_user_study(complete_df)
# load
class_step.load_models()
train_df, _, _ = class_step.load_dfs()
opt_step = OptimizationStep(class_step)
print(len(complete_df) == len(train_df))
sim_obj = Simulation(class_step, opt_step)

cost_opt, claims = sim_obj.get_cost_random_order_opt(test_df)
print("cost from random order using opt: ", cost_opt)

# cost_ver_only, claims = sim_obj.get_cost_random_order_only_verification(test_df)
# print("cost from random order using verification only: ", cost_ver_only)

stats_dict = dict()
topn = 5
for task_name, task in class_step.classification_tasks_dict.items():
    topn = task.topn
    stats_dict[task_name] = [0]*(task.topn+1)
stats_dict["all_props"] = [0]*(topn+1)

get_stats(claims, stats_dict)