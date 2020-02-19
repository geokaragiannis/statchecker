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
        self.init_number_formulas = self.get_init_number_formulas()

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
        number_remaining_formulas = self.init_number_formulas
        idx = 0
        # print("statring props to ask")
        # print("num properties questions: ", num_properties_questions)
        if num_properties_questions > len(claim.available_properties) - 1:
            # ask about all claims (minus the template formula)
            num_to_ask = len(claim.available_properties) - 1
        else:
            num_to_ask = num_properties_questions

        # print("num to ask: ", num_to_ask)
        for i in range(num_to_ask):
            max_factor_property = None
            max_prunning_factor = 0
            for prop in claim.available_properties:
                if prop.property_name == "template_formula":
                    prop.ask = True
                    continue
                idx += 1
                if not prop.ask:
                    prop.ask = True
                    prunning_factor = self.get_number_formulas_excluded_per_var(number_remaining_formulas, claim)
                    # print("property: {}, prunning factor: {}".format(prop.property_name, prunning_factor))
                    prop.ask = False
                    if prunning_factor > max_prunning_factor:
                        max_prunning_factor = prunning_factor
                        max_factor_property = prop

            max_factor_property.ask = True
            number_remaining_formulas = number_remaining_formulas - max_prunning_factor
            print("step: {}, max_factor_prop: {}, remaining formulas: {}".format(i, max_factor_property.property_name, number_remaining_formulas))



    def get_number_formulas_excluded_per_var(self, number_remaining_formulas, claim):
        """
        Return the number of formulas a specific value for a property (i.e TPEDTOTAL)
        excludes per variable. I.e the number of formulas that does not have as row_index TPEDTOTAL
        Arguments:
            number_remaining_formulas {int} -- [contains the number of times each property value occurs]
            property_value {str} -- [TPEDTOTAL or file1 or ...]
            property_name {str} -- [row_index or file or column]
        """

        return number_remaining_formulas - self.get_remaining_number_formulas(claim)



    def get_remaining_number_formulas(self, claim):
        """
        Get the remaining number of total formulas per variable after picking
        some properties already
        
        Arguments:
            properties_picked {list of str} -- [description]
        """

        remaining_properties = []
        file_asked = True
        row_not_asked = False
        file_top_pred = None
        for prop in claim.available_properties:
            if prop.property_name == "template_formula":
                continue
            if prop.property_name == "file":
                file_top_pred = prop.candidate_values[0].value
            if not prop.ask:
                remaining_properties.append(prop)
                if prop.property_name == "row_index":
                    row_not_asked = True
                if prop.property_name == "file":
                    file_asked = False

        # if we have selected a file and not a row_idx use the contraints
        use_constraints = file_asked and row_not_asked

        # print("remaining properties: ", [prop.property_name for prop in remaining_properties])


        remaining_num = 1
        for rem_prop in remaining_properties:
            if use_constraints and rem_prop.property_name == "row_index":
                # get number of possible rows from constraint dict. If it fails consider all the rows
                try:
                    file_to_row_constraint_dict = helpers.load_contraint_file("file2row_index.json")
                    num_all_values = file_to_row_constraint_dict.get[file_top_pred]
                except:
                    num_all_values = len(self.classification_step.classification_tasks_dict[rem_prop.property_name].all_values)
            else:
                num_all_values = len(self.classification_step.classification_tasks_dict[rem_prop.property_name].all_values)
            remaining_num *= num_all_values 

        return remaining_num


    def get_init_number_formulas(self):
        """
        Get the inital number of all template formulas for one variable
        """
        # no contraints
        all_files = self.classification_step.classification_tasks_dict["file"].all_values
        all_row_indexes = self.classification_step.classification_tasks_dict["row_index"].all_values
        all_columns = self.classification_step.classification_tasks_dict["column"].all_values
        file_to_row_constraint_dict = helpers.load_contraint_file("file2row_index.json")
        # print("sizes of files: {}, row: {}, col: {} ".format(len(all_files), len(all_row_indexes), len(all_columns)) )
        # get all rows available from the constraint
        all_files_rows = 0
        for file in all_files:
            num_rows_from_file = len(file_to_row_constraint_dict.get(file, all_row_indexes))
            all_files_rows += num_rows_from_file

        return all_files_rows * len(all_columns)

