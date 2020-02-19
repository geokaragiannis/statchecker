"""
This class receives a claim object without predictions and a classification_step object
which contains the necessary featurizers and classifiers for prediction. Here we:
1. Compute the number of values to display per property
2. Compute the number of properties to ask about
3. Get predictions for the claim ???
4. Return the subset of properties we ask about along with their proposed values

!!!! Note that we expect that we have trained our model !!!!
"""
from src import helpers
from src.crowdsourcing.claim import Claim
from src.crowdsourcing.property import Property
from src.crowdsourcing.value import Value

import math
from collections import Counter
import numpy as np


class OptimizationStep:
    def __init__(self, classification_step):
        self.classification_step = classification_step
        self.max_num_prop = None
        self.all_formula_var_instances_dict = {"file": Counter(), "row_index": Counter(), "column": Counter()}
        self.get_all_instances_of_var()
        if classification_step.train_df is not None:
            classification_step.parser.set_task_values(classification_step.train_df)
        else:
            complete_df = classification_step.parser.get_complete_df()
            classification_step.parser.set_task_values(complete_df)

    def optimize_claim(self, claim):

        self.get_topn(claim)
        max_properties_questions = self.get_max_num_properties_to_ask(lamda=4)
        self.get_preds_from_claim(claim)
        if max_properties_questions < len(claim.available_properties):
            num_properties_questions = max_properties_questions 
        else:
            num_properties_questions = len(claim.available_properties)
        self.get_subset_props_to_ask(claim, num_properties_questions)
        

    def get_preds_from_claim(self, claim):

        for prop in claim.available_properties:
            featurizer_tf = prop.task.featurizer_tf
            featurizer_emb = prop.task.featurizer_emb
            classifier = prop.task.classifier
            features = self.classification_step.get_feature_union([claim.sent], [claim.claim], self.classification_step.tok_driver, 
                                                           featurizer_emb, featurizer_tf, mode="test")
            pred_labels, pred_probs = classifier.predict_utt_top_n(features, n=prop.task.topn)
            if prop.property_name == "template_formula":
                values_list = [Value(label, prob, prop) for label, prob in zip(pred_labels, pred_probs)]
            else:
                hash_to_label_dict = prop.task.hash_to_label_dict
                try:
                    values_list = [Value(hash_to_label_dict[str(int(label))], prob, prop) for label, prob in zip(pred_labels, pred_probs)]
                except: 
                    values_list = [Value(label, prob, prop) for label, prob in zip(pred_labels, pred_probs)]
            prop.candidate_values = values_list

    def get_topn(self, claim):
        """
        For each property, compute the number of options to display
        """
        for task in self.classification_step.classification_tasks_dict.values():
            task.topn = math.floor(task.der_cost/task.ver_cost)
    
    def get_max_num_properties_to_ask(self, lamda=4):
        """
        Get the maximum number of properties we can ask about. I.e if the max
        number of properties is 2, we can ask about e.g file, row only.
        lamda is a constant. The higher it is, the more questions we can ask
        """
        # assuming derivation cost of all properties except template_formula is the same
        der_cost_prop = self.classification_step.classification_tasks_dict["row_index"].der_cost
        der_cost_formula = self.classification_step.classification_tasks_dict["template_formula"].der_cost
            
        return math.floor(der_cost_formula*(lamda-2)/(2*der_cost_prop))

    
    def get_subset_props_to_ask(self, claim, num_properties_questions):
        """
        Populate the "ask" field of the claim's properties an ordered list of properties to ask.
        Here we assume that we have the classifiers predictions on the claims. 
        Arguments:
            claim {Claim obj} -- [The claim for which we wish to know the prperties to ask]
        """
        all_formulas_dict = self.all_formula_var_instances_dict
        idx = 0
        for prop in claim.available_properties:
            if prop.property_name == "template_formula":
                continue
            idx += 1
            if idx == num_properties_questions:
                break
            max_prunning_factor = np.inf
            if not prop.ask:
                top_pred_value = prop.candidate_values[0]
                prunning_factor = self.get_number_formulas_excluded_per_var(all_formulas_dict, top_pred_value.value, prop.property_name)
                if prunning_factor < max_prunning_factor:
                    max_prunning_factor = prunning_factor
                    prop.ask = True



    def get_number_formulas_excluded_per_var(self, all_formulas, property_value, property_name):
        """
        Return the number of formulas a specific value for a property (i.e TPEDTOTAL)
        excludes per variable. I.e the number of formulas that does not have as row_index TPEDTOTAL
        Arguments:
            all_formulas {dict of Counters} -- [contains the number of times each property value occurs]
            property_value {str} -- [TPEDTOTAL or file1 or ...]
            property_name {str} -- [row_index or file or column]
        """
        num_occurences = all_formulas[property_name][property_value]
        total_num_formulas = 0
        for _, counter_dict in all_formulas.items():
            total_num_formulas += sum(counter_dict.values())
        
        return total_num_formulas - num_occurences

    
    # def update_all_instances_of_var(self, property_answered):
    #     # if property_answered == "file":
    #     #     # get rows from constraint dict
    #     remaining_properties = set(["file", "row_index", "column"]) - set(property_answered)


    def get_all_instances_of_var(self):
        """
        Return a a counter dict, which contains how many times a sprcific value of a property
        occurs in all the possible instances of a variable. E.g we return a dict which looks like:
        {"TPEDTOTAL": 100, "file2": 10221, "2012": 10, ...}
        Here we only use file, row_index and column.
        """
        # TODO: fix the file2row_index constraint to have the same keys as the data we train on
        # sets of all the properties we consider
        all_files = self.classification_step.classification_tasks_dict["file"].all_values
        all_row_indexes = self.classification_step.classification_tasks_dict["row_index"].all_values
        all_columns = self.classification_step.classification_tasks_dict["column"].all_values
        # file_to_row_constraint_dict = helpers.load_contraint_file("file2row_index.json")
        # print("aaa \n\n\n\n\n\n", file_to_row_constraint_dict.keys())
       
        for file in all_files:
            for row in all_row_indexes:
                for column in all_columns:
                    # d = {"file": file, "row_index": row, "column": column}
                    # all_formula_var_instances.append(d)
                    self.all_formula_var_instances_dict["file"][file] += 1
                    self.all_formula_var_instances_dict["row_index"][row] += 1
                    self.all_formula_var_instances_dict["column"][column] += 1

        return self.all_formula_var_instances_dict

