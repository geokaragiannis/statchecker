"""
Includes several simulation experiments
"""
DATA_PATH = "data/claims_02-19-2020/"

import pandas as pd
from src.pipeline.classification_step import ClassificationStep
from src.pipeline.active_learning_step import ActiveLearningStep
from src.pipeline.optimization_step import OptimizationStep
from src.crowdsourcing.simulation import Simulation
import json
import time

def get_stats(claims, stats_dict):
    for claim in claims:
        for prop in claim.available_properties:
            stats_dict[prop.property_name]["num"][prop.verified_index] += 1
            if prop.verified_index == -1:
                for i in range(len(prop.candidate_values)):
                    stats_dict[prop.property_name]["mean_prob_der"][i] += prop.candidate_values[i].prob/len(claims)
            else:
                for i in range(len(prop.candidate_values)):
                    stats_dict[prop.property_name]["mean_prob_ver"][i] += prop.candidate_values[i].prob/len(claims)
                
            # stats_dict["all_props"][prop.verified_index] += 1
    print(json.dumps(stats_dict))


def export_claims_preds(claims_list):
    export_list = []
    for claim in claims_list:
        export_list.append({"cost": claim.real_cost, "avg_property_classifier_acc": claim.avg_class_accuracy})
    df = pd.DataFrame(export_list)
    df.to_csv("data/milp_acc_vs_cost_all_claims.csv")

def cold_exp_sequential(complete_df, class_step):
    print("SEQUENTIAL\n")
    complete_df = complete_df.sort_values(by="subsection")
    print("len of complete df: ", len(complete_df))
    train_df = complete_df.iloc[:10]
    test_df = complete_df[10:]
    batch_idx_list = [90, 190, 290, 390, 490, 590, 690, 790, 890, 990, 1090, 1190, 1290, 1390, 1490, len(test_df)]
    start_time = time.time()
    class_step.train_for_user_study(train_df)
    opt_step = OptimizationStep(class_step)
    active_step = ActiveLearningStep(class_step)
    sim_obj = Simulation(class_step, opt_step, active_step)
    cost, claims = sim_obj.get_cost_sequential_order_opt_retraining(train_df, test_df, batch_idx_list)
    print("cost from sequential order using opt and retraining: ", cost)
    end_time = time.time()
    print("\nOVERALL TIME: ", end_time-start_time)


def cold_exp_active_learning_milp(complete_df, class_step):
    print("MILP\n")
    complete_df = complete_df.sort_values(by="subsection")
    print("len of complete df: ", len(complete_df))
    train_df = complete_df.iloc[:10]
    test_df = complete_df[10:]
    batch_idx_list = [90, 190, 290, 390, 490, 590, 690, 790, 890, 990, 1090, 1190, 1290, 1390, 1490, len(test_df)]
    start_time = time.time()
    class_step.train_for_user_study(train_df)
    opt_step = OptimizationStep(class_step)
    active_step = ActiveLearningStep(class_step)
    sim_obj = Simulation(class_step, opt_step, active_step)
    skim_cost = 1
    print("skim cost: ", skim_cost)
    cost, claims = sim_obj.get_cost_active_learning_milp_opt_retraining(train_df, test_df, batch_idx_list, skim_cost=skim_cost)
    print("cost from milp using opt and retraining: ", cost)
    end_time = time.time()
    print("\nOVERALL TIME: ", end_time-start_time)
    export_claims_preds(claims)


class_step = ClassificationStep(DATA_PATH, simulation=False, export=False)
complete_df = class_step.parser.get_complete_df()


# cold_exp_sequential(complete_df, class_step)
cold_exp_active_learning_milp(complete_df, class_step)